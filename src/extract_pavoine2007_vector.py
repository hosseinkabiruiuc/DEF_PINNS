from __future__ import annotations
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import pdfplumber
ROOT = Path(__file__).resolve().parents[1]
PDF = next((ROOT / 'data/source_pdfs').glob('DEF-000170_*.pdf'))
OUT = ROOT / 'data/extracted'
PLOTS = {'Pavoine2007_Fig1': {'page': 6, 'rough': (40, 78, 270, 190), 'temperature_C': 65.0, 'aggregate_type': 'siliceous', 'design': 'heating'}, 'Pavoine2007_Fig2': {'page': 6, 'rough': (40, 255, 270, 365), 'temperature_C': 85.0, 'aggregate_type': 'siliceous', 'design': 'heating'}, 'Pavoine2007_Fig3': {'page': 6, 'rough': (40, 425, 270, 540), 'temperature_C': 85.0, 'aggregate_type': 'siliceous', 'design': 'concrete_mix'}, 'Pavoine2007_Fig4': {'page': 6, 'rough': (280, 70, 500, 185), 'temperature_C': 85.0, 'aggregate_type': 'calcareous', 'design': 'concrete_mix'}}
DURATION = {'2h': 2.0 / 24.0, '6h': 6.0 / 24.0, '2d': 2.0, '10d': 10.0}

def color_mean(value: Any) -> float:
    if isinstance(value, (tuple, list)) and value:
        return float(sum(value) / len(value))
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan

def duration_from_style(obj: dict[str, Any], outline_stroke: float) -> str | None:
    fill = color_mean(obj.get('non_stroking_color'))
    if math.isnan(fill) or math.isnan(outline_stroke):
        return None
    if fill >= 0.95:
        return '2h'
    if fill <= 0.2:
        return '10d'
    if outline_stroke >= 0.7:
        return '6h'
    if outline_stroke < 0.7:
        return '2d'
    return None

def shape_from_object(kind: str, obj: dict[str, Any]) -> str | None:
    if kind == 'rect':
        return 'square'
    points = len(obj.get('pts') or [])
    if kind == 'curve' and points in (5, 6):
        return 'circle'
    if kind == 'curve' and points == 4:
        return 'triangle'
    return None

def numeric_word(word: dict[str, Any]) -> float | None:
    value = word['text'].replace(',', '.').strip()
    if not re.fullmatch('-?\\d+(?:\\.\\d+)?', value):
        return None
    try:
        return float(value)
    except ValueError:
        return None

def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    n = len(points)
    sx = sum((x for x, _ in points))
    sy = sum((y for _, y in points))
    sxx = sum((x * x for x, _ in points))
    sxy = sum((x * y for x, y in points))
    denominator = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / n
    mean_y = sy / n
    ss_tot = sum(((y - mean_y) ** 2 for _, y in points))
    ss_res = sum(((y - (slope * x + intercept)) ** 2 for x, y in points))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return (slope, intercept, r2)

def calibrate(words: list[dict[str, Any]], rough: tuple[float, float, float, float]) -> dict[str, float]:
    left, top, right, bottom = rough
    x_ticks: list[tuple[float, float]] = []
    y_ticks: list[tuple[float, float]] = []
    for word in words:
        value = numeric_word(word)
        if value is None:
            continue
        cx = (word['x0'] + word['x1']) / 2.0
        cy = (word['top'] + word['bottom']) / 2.0
        if left + 15 <= cx <= right - 10 and bottom - 22 <= cy <= bottom + 4 and (value in {50, 150, 250, 350, 450, 550, 650}):
            x_ticks.append((cx, value))
        if left + 8 <= cx <= left + 32 and top - 5 <= cy <= bottom - 12 and (word['text'].strip() in {'0,0', '0,2', '0,4', '0,6', '0,8', '1,0', '1,2'}):
            y_ticks.append((cy, value))
    x_ticks = sorted(set(((round(x, 3), y) for x, y in x_ticks)))
    y_ticks = sorted(set(((round(x, 3), y) for x, y in y_ticks)))
    if len(x_ticks) < 4 or len(y_ticks) < 5:
        raise ValueError(f'insufficient axis ticks: x={x_ticks}, y={y_ticks}')
    tx_slope, tx_intercept, x_r2 = linear_fit(x_ticks)
    ey_slope, ey_intercept, y_r2 = linear_fit(y_ticks)
    x_min = (50.0 - tx_intercept) / tx_slope
    x_max = (650.0 - tx_intercept) / tx_slope
    y_bottom = (0.0 - ey_intercept) / ey_slope
    y_top = (1.2 - ey_intercept) / ey_slope
    return {'time_slope': tx_slope, 'time_intercept': tx_intercept, 'expansion_slope': ey_slope, 'expansion_intercept': ey_intercept, 'x_r2': x_r2, 'y_r2': y_r2, 'x_min': min(x_min, x_max), 'x_max': max(x_min, x_max), 'y_top': min(y_top, y_bottom), 'y_bottom': max(y_top, y_bottom), 'x_tick_count': len(x_ticks), 'y_tick_count': len(y_ticks)}

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    with pdfplumber.open(PDF) as document:
        for figure_id, spec in PLOTS.items():
            page = document.pages[int(spec['page']) - 1]
            calibration = calibrate(page.extract_words(), spec['rough'])
            calibration_rows.append({'figure_id': figure_id, **calibration})
            grouped: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
            outline_strokes: dict[tuple[str, float, float], float] = {}
            for kind, objects in (('curve', page.curves), ('rect', page.rects)):
                for obj in objects:
                    if obj.get('fill') or not obj.get('stroke'):
                        continue
                    width = float(obj['x1'] - obj['x0'])
                    height = float(obj['bottom'] - obj['top'])
                    if not (2.5 <= width <= 3.2 and 2.5 <= height <= 3.2):
                        continue
                    cx = round((obj['x0'] + obj['x1']) / 2.0, 2)
                    cy = round((obj['top'] + obj['bottom']) / 2.0, 2)
                    outline_strokes[kind, cx, cy] = color_mean(obj.get('stroking_color'))
            for kind, objects in (('curve', page.curves), ('rect', page.rects)):
                for obj in objects:
                    if not obj.get('fill'):
                        continue
                    width = float(obj['x1'] - obj['x0'])
                    height = float(obj['bottom'] - obj['top'])
                    if not (2.5 <= width <= 3.2 and 2.5 <= height <= 3.2):
                        continue
                    cx = (obj['x0'] + obj['x1']) / 2.0
                    cy = (obj['top'] + obj['bottom']) / 2.0
                    if not calibration['x_min'] - 2 <= cx <= calibration['x_max'] + 2:
                        continue
                    if not calibration['y_top'] - 2 <= cy <= calibration['y_bottom'] + 2:
                        continue
                    outline_stroke = outline_strokes.get((kind, round(cx, 2), round(cy, 2)), math.nan)
                    duration = duration_from_style(obj, outline_stroke)
                    shape = shape_from_object(kind, obj)
                    if duration is None or shape is None:
                        continue
                    if spec['design'] == 'heating' and shape not in {'square', 'circle', 'triangle'}:
                        continue
                    if spec['design'] == 'concrete_mix' and shape not in {'square', 'circle'}:
                        continue
                    time_days = calibration['time_slope'] * cx + calibration['time_intercept']
                    expansion = calibration['expansion_slope'] * cy + calibration['expansion_intercept']
                    grouped[shape, duration].append((time_days, expansion, cx, cy))
            for (shape, duration), points in grouped.items():
                deduped: list[tuple[float, float, float, float]] = []
                for time_days, expansion, source_x, source_y in sorted(points):
                    if deduped and abs(time_days - deduped[-1][0]) < 0.35 and (abs(expansion - deduped[-1][1]) < 0.003):
                        continue
                    deduped.append((time_days, expansion, source_x, source_y))
                if len(deduped) < 6:
                    continue
                alkali = 0.75
                water_cement = 0.48
                if spec['design'] == 'heating':
                    alkali = {'square': 0.5, 'circle': 0.75, 'triangle': 1.0}[shape]
                else:
                    water_cement = {'square': 0.48, 'circle': 0.35}[shape]
                curve_id = f'{figure_id}_{shape}_{duration}'
                for point_index, (time_days, expansion, source_x, source_y) in enumerate(deduped, start=1):
                    rows.append({'record_id': 'DEF-000170', 'source_short': 'Brunetaud et al. 2007', 'DOI': '10.1617/s11527-006-9163-3', 'page_number': 6, 'figure_id': figure_id, 'curve_id': curve_id, 'point_index': point_index, 'time_days': round(time_days, 3), 'expansion_percent': round(expansion, 6), 'source_pdf_x': round(source_x, 3), 'source_pdf_top': round(source_y, 3), 'temperature_C': spec['temperature_C'], 'treatment_duration_days': DURATION[duration], 'alkali_percent': '' if alkali is None else alkali, 'water_cement_ratio': water_cement, 'aggregate_type': spec['aggregate_type'], 'moisture_condition': '20 C, 100% RH after cyclic wetting/drying', 'scm_type': 'none', 'scm_percent': 0.0, 'extraction_method': 'vector_marker_geometry', 'axis_calibration_r2_time': round(calibration['x_r2'], 8), 'axis_calibration_r2_expansion': round(calibration['y_r2'], 8), 'quality_status': 'accepted' if min(calibration['x_r2'], calibration['y_r2']) >= 0.997 else 'review'})
    fields = list(rows[0].keys())
    with (OUT / 'pavoine2007_vector_curves.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    calibration_fields = list(calibration_rows[0].keys())
    with (OUT / 'pavoine2007_axis_calibration.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=calibration_fields)
        writer.writeheader()
        writer.writerows(calibration_rows)
    curve_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        curve_counts[str(row['curve_id'])] += 1
    summary = {'source': 'Brunetaud et al. 2007', 'record_id': 'DEF-000170', 'figures': len(PLOTS), 'curves': len(curve_counts), 'points': len(rows), 'minimum_points_per_curve': min(curve_counts.values()), 'maximum_points_per_curve': max(curve_counts.values()), 'all_calibrations_r2_at_least_0_997': all((min(float(row['x_r2']), float(row['y_r2'])) >= 0.997 for row in calibration_rows))}
    (OUT / 'pavoine2007_extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
