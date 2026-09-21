#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_annotations.py — универсальная проверка разметки фото.

Поддерживает:
  - COCO JSON (bbox, polygon через segmentation)
  - CVAT for images 1.1 (XML): box, polygon, points

Проверяет:
  - структуру файла
  - выход координат за границы изображения
  - замкнутость полигонов
  - самопересечения полигонов
  - нулевую/отрицательную площадь
  - наличие аннотаций у каждого изображения
  - совпадение классов с ожидаемыми (опционально)
  - дубли аннотаций (опционально)

Запуск:
  python check_annotations.py --input annotations/photo_bbox.json
  python check_annotations.py --input annotations/car_polygon.xml
  python check_annotations.py --input annotations/photo_bbox.json --classes dog cat car
  python check_annotations.py --input annotations/car_polygon.xml --strict
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

EPS = 1e-9


# ----------------------------- утилиты -----------------------------

def seg_intersect(p1, p2, p3, p4) -> bool:
    """Пересекаются ли отрезки p1-p2 и p3-p4 (без учёта общих концов)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS and
                min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS)

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and \
       ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)):
        return True

    if abs(d1) < EPS and on_segment(p3, p4, p1):
        return True
    if abs(d2) < EPS and on_segment(p3, p4, p2):
        return True
    if abs(d3) < EPS and on_segment(p1, p2, p3):
        return True
    if abs(d4) < EPS and on_segment(p1, p2, p4):
        return True

    return False


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Площадь полигона по формуле шнуровки (абсолютная)."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def is_closed(points: List[Tuple[float, float]]) -> bool:
    """Замкнут ли контур: первая и последняя точки совпадают."""
    if len(points) < 3:
        return False
    return abs(points[0][0] - points[-1][0]) < EPS and \
           abs(points[0][1] - points[-1][1]) < EPS


def has_self_intersections(points: List[Tuple[float, float]],
                           closed: bool = True) -> List[Tuple[int, int]]:
    """Возвращает список пар индексов рёбер, которые пересекаются."""
    pts = list(points)
    if closed and not is_closed(pts):
        pts = pts + [pts[0]]

    n = len(pts)
    if n < 4:
        return []

    intersections = []
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            # соседние рёбра не считаем
            if j == i + 1:
                continue
            # первое и последнее ребро — соседи при замкнутом контуре
            if closed and i == 0 and j == n - 2:
                continue
            if seg_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                intersections.append((i, j))
    return intersections


def bbox_from_points(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_image(x: float, y: float, w: int, h: int) -> bool:
    return -EPS <= x <= w + EPS and -EPS <= y <= h + EPS


def bbox_in_image(x: float, y: float, bw: float, bh: float, w: int, h: int) -> bool:
    return (x >= -EPS and y >= -EPS and
            x + bw <= w + EPS and y + bh <= h + EPS and
            bw > EPS and bh > EPS)


# ----------------------------- отчёт -----------------------------

class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def note(self, msg):
        self.info.append(msg)

    def print(self):
        print("ОТЧЁТ О ПРОВЕРКЕ")

        if self.info:
            print("\n[ИНФО]")
            for m in self.info:
                print(f"  • {m}")

        if self.warnings:
            print("\n[ПРЕДУПРЕЖДЕНИЯ]")
            for m in self.warnings:
                print(f"  ! {m}")

        if self.errors:
            print("\n[ОШИБКИ]")
            for m in self.errors:
                print(f"  ✗ {m}")
        else:
            print("\n[ОШИБКИ] не найдены")
        print(f"Итого: ошибок — {len(self.errors)}, "
              f"предупреждений — {len(self.warnings)}")

        return len(self.errors) == 0


# ----------------------------- COCO -----------------------------

def check_coco(path: Path, expected_classes: Optional[List[str]],
               strict: bool) -> Report:
    rep = Report()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        rep.error(f"Не удалось прочитать JSON: {e}")
        return rep

    for key in ("images", "annotations", "categories"):
        if key not in data:
            rep.error(f"В файле нет обязательного блока '{key}'")
    if rep.errors:
        return rep

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    annotations = data["annotations"]

    rep.note(f"Изображений: {len(images)}")
    rep.note(f"Аннотаций: {len(annotations)}")
    rep.note(f"Классов: {len(categories)}")

    if expected_classes:
        actual = set(categories.values())
        expected = set(expected_classes)
        missing = expected - actual
        extra = actual - expected
        if missing:
            rep.error(f"В файле нет ожидаемых классов: {sorted(missing)}")
        if extra:
            rep.warn(f"Лишние классы в файле: {sorted(extra)}")

    # у каждого изображения должна быть хотя бы одна аннотация
    ann_by_image: Dict[int, int] = {img_id: 0 for img_id in images}
    seen_bboxes = {}

    for i, ann in enumerate(annotations):
        ann_id = ann.get("id", i)
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")

        if img_id not in images:
            rep.error(f"Аннотация id={ann_id}: image_id={img_id} не найден")
            continue
        if cat_id not in categories:
            rep.error(f"Аннотация id={ann_id}: category_id={cat_id} не найден")
            continue

        ann_by_image[img_id] += 1
        img = images[img_id]
        w, h = img.get("width"), img.get("height")
        if not w or not h:
            rep.warn(f"Изображение id={img_id}: нет width/height, "
                     f"проверка границ пропущена")
            w = h = None

        # bbox
        bbox = ann.get("bbox")
        if bbox is not None:
            if len(bbox) != 4:
                rep.error(f"Аннотация id={ann_id}: bbox должен быть [x,y,w,h]")
            else:
                x, y, bw, bh = bbox
                if bw <= EPS or bh <= EPS:
                    rep.error(f"Аннотация id={ann_id}: нулевая/отрицательная "
                              f"площадь bbox ({bw}×{bh})")
                if w and h and not bbox_in_image(x, y, bw, bh, w, h):
                    rep.error(f"Аннотация id={ann_id}: bbox выходит за границы "
                              f"изображения {w}×{h}")

            # дубли bbox
            key = (img_id, cat_id, tuple(round(v, 2) for v in bbox))
            if key in seen_bboxes:
                rep.warn(f"Аннотация id={ann_id}: возможный дубль bbox "
                         f"(совпадает с id={seen_bboxes[key]})")
            else:
                seen_bboxes[key] = ann_id

        # segmentation (polygon)
        seg = ann.get("segmentation")
        if seg:
            polygons = seg if isinstance(seg[0], list) else [seg]
            for pi, poly in enumerate(polygons):
                if len(poly) < 6:
                    rep.error(f"Аннотация id={ann_id}, полигон {pi}: "
                              f"меньше 3 точек")
                    continue
                pts = [(poly[k], poly[k + 1]) for k in range(0, len(poly), 2)]
                _check_polygon(rep, ann_id, pi, pts, w, h)

    # изображения без аннотаций
    empty = [img_id for img_id, cnt in ann_by_image.items() if cnt == 0]
    if empty:
        rep.error(f"Изображения без аннотаций: {empty}")

    return rep


def _check_polygon(rep: Report, ann_id: int, pi: int,
                   pts: List[Tuple[float, float]],
                   w: Optional[int], h: Optional[int]):
        # замкнутость
    # Замыкание подразумевается неявно
    # если контур замкнут, но при этом меньше 3 уникальных точек
    # (это уже признак ошибки).
    if is_closed(pts):
        unique = set(pts[:-1]) if len(pts) > 1 else set(pts)
        if len(unique) < 3:
            rep.error(f"Аннотация id={ann_id}, полигон {pi}: "
                      f"меньше 3 уникальных точек")
    # площадь
    area = polygon_area(pts if is_closed(pts) else pts + [pts[0]])
    if area <= EPS:
        rep.error(f"Аннотация id={ann_id}, полигон {pi}: нулевая площадь")

    # границы
    if w and h:
        for k, (x, y) in enumerate(pts):
            if not point_in_image(x, y, w, h):
                rep.error(f"Аннотация id={ann_id}, полигон {pi}, точка {k}: "
                          f"({x:.1f}, {y:.1f}) выходит за границы {w}×{h}")
                break

    # самопересечения
    inter = has_self_intersections(pts, closed=is_closed(pts))
    if inter:
        rep.error(f"Аннотация id={ann_id}, полигон {pi}: самопересечения "
                  f"в рёбрах {inter}")


# ----------------------------- CVAT XML -----------------------------

def check_cvat_xml(path: Path, expected_classes: Optional[List[str]],
                   strict: bool) -> Report:
    rep = Report()
    try:
        tree = ET.parse(path)
    except Exception as e:
        rep.error(f"Не удалось прочитать XML: {e}")
        return rep

    root = tree.getroot()

        # meta -> labels
    # CVAT хранит классы в разных местах в зависимости от версии
    # и от того, экспортировался проект или задача:
    #   meta/project/labels/label   (экспорт проекта)
    #   meta/task/labels/label      (экспорт задачи)
    #   meta/job/labels/label       (экспорт job)
    labels = []
    for path in (
        ".//meta/project/labels/label",
        ".//meta/task/labels/label",
        ".//meta/job/labels/label",
    ):
        for label_el in root.findall(path):
            name_el = label_el.find("name")
            if name_el is not None:
                labels.append(name_el.text)

    rep.note(f"Классов в meta: {len(labels)}")
    if expected_classes:
        missing = set(expected_classes) - set(labels)
        extra = set(labels) - set(expected_classes)
        if missing:
            rep.error(f"В meta нет ожидаемых классов: {sorted(missing)}")
        if extra:
            rep.warn(f"Лишние классы в meta: {sorted(extra)}")

    images = root.findall(".//image")
    if not images:
        rep.error("В файле нет ни одного <image>")
        return rep

    total_boxes = total_polygons = total_points = 0

    for img in images:
        img_name = img.get("name", "?")
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        if not w or not h:
            rep.warn(f"Изображение '{img_name}': нет width/height, "
                     f"проверка границ пропущена")

        for el in img:
            tag = el.tag
            label = el.get("label", "?")

            if tag == "box":
                total_boxes += 1
                try:
                    xtl = float(el.get("xtl"))
                    ytl = float(el.get("ytl"))
                    xbr = float(el.get("xbr"))
                    ybr = float(el.get("ybr"))
                except (TypeError, ValueError):
                    rep.error(f"'{img_name}' box label='{label}': "
                              f"некорректные координаты")
                    continue
                bw, bh = xbr - xtl, ybr - ytl
                if bw <= EPS or bh <= EPS:
                    rep.error(f"'{img_name}' box label='{label}': "
                              f"нулевая/отрицательная площадь")
                if w and h and not bbox_in_image(xtl, ytl, bw, bh, w, h):
                    rep.error(f"'{img_name}' box label='{label}': "
                              f"выходит за границы {w}×{h}")

            elif tag == "polygon":
                total_polygons += 1
                pts = _parse_points(el.get("points", ""))
                if len(pts) < 3:
                    rep.error(f"'{img_name}' polygon label='{label}': "
                              f"меньше 3 точек")
                    continue
                _check_polygon(rep, f"'{img_name}'", label, pts, w, h)

            elif tag == "points":
                total_points += 1
                pts = _parse_points(el.get("points", ""))
                if not pts:
                    rep.error(f"'{img_name}' points label='{label}': "
                              f"нет координат")
                    continue
                if w and h:
                    for k, (x, y) in enumerate(pts):
                        if not point_in_image(x, y, w, h):
                            rep.error(f"'{img_name}' points label='{label}', "
                                      f"точка {k}: выходит за границы")
                            break

    rep.note(f"Всего изображений: {len(images)}")
    rep.note(f"Всего box: {total_boxes}")
    rep.note(f"Всего polygon: {total_polygons}")
    rep.note(f"Всего points: {total_points}")

    return rep


def _parse_points(s: str) -> List[Tuple[float, float]]:
    """'x1,y1;x2,y2;...' -> [(x1,y1), ...]"""
    pts = []
    if not s:
        return pts
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x, y = pair.split(",")
            pts.append((float(x), float(y)))
        except ValueError:
            continue
    return pts


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Проверка разметки фото (COCO JSON / CVAT XML)")
    parser.add_argument("--input", "-i", required=True,
                        help="Путь к файлу разметки (.json или .xml)")
    parser.add_argument("--classes", "-c", nargs="*", default=None,
                        help="Ожидаемые классы, например: dog cat car")
    parser.add_argument("--strict", action="store_true",
                        help="Считать предупреждения ошибками")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Файл не найден: {path}")
        sys.exit(2)

    suffix = path.suffix.lower()
    if suffix == ".json":
        rep = check_coco(path, args.classes, args.strict)
    elif suffix == ".xml":
        rep = check_cvat_xml(path, args.classes, args.strict)
    else:
        print(f"Неподдерживаемый формат: {suffix}")
        sys.exit(2)

    ok = rep.print()

    if args.strict and rep.warnings:
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()