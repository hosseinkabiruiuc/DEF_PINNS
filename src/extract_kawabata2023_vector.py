from __future__ import annotations
import csv
import json
from pathlib import Path
import pdfplumber
ROOT = Path(__file__).resolve().parents[1]
PDF = next((ROOT / 'data/source_pdfs').glob('DEF-001010_*.pdf'))
OUT = ROOT / 'data/extracted'
COLORS = {'black': (0.0, 0.0, 0.0), 'red': (1.0, 0.0, 0.0), 'blue': (0.0, 0.0, 1.0), 'yellow': (1.0, 0.753, 0.0), 'green': (0.0, 1.0, 0.0), 'magenta': (1.0, 0.0, 1.0)}
PANELS = [{'figure': 'Fig5a', 'stirrups': False, 'direction': 'longitudinal', 'x_bounds': (96, 280), 'y_bounds': (376, 505), 'x_tick_y': 511.4, 'y_tick_x': 83.8, 'positions': {'black': 100, 'red': 200, 'blue': 300, 'yellow': 400, 'green': 500}}, {'figure': 'Fig5b', 'stirrups': False, 'direction': 'transverse', 'x_bounds': (336, 520), 'y_bounds': (376, 505), 'x_tick_y': 511.4, 'y_tick_x': 325.9, 'positions': {'black': 50, 'red': 150, 'blue': 250, 'yellow': 350, 'green': 450, 'magenta': 550}}, {'figure': 'Fig6a', 'stirrups': True, 'direction': 'longitudinal', 'x_bounds': (95, 280), 'y_bounds': (588, 716), 'x_tick_y': 722.7, 'y_tick_x': 82.8, 'positions': {'black': 100, 'red': 200, 'blue': 300, 'yellow': 400, 'green': 500}}, {'figure': 'Fig6b', 'stirrups': True, 'direction': 'transverse', 'x_bounds': (336, 521), 'y_bounds': (588, 716), 'x_tick_y': 722.7, 'y_tick_x': 325.4, 'positions': {'black': 50, 'red': 150, 'blue': 250, 'yellow': 350, 'green': 450, 'magenta': 550}}]

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
    return (slope, intercept, 1.0 - ss_res / ss_tot)

def calibrate(page: pdfplumber.page.Page, panel: dict[str, object]) -> tuple[float, float, float, float, float, float]:
    x_ticks = []
    y_ticks = []
    for word in page.extract_words():
        cx = (word['x0'] + word['x1']) / 2
        cy = (word['top'] + word['bottom']) / 2
        if word['text'] in {'0', '50', '100', '150', '200', '250'} and abs(cy - float(panel['x_tick_y'])) < 6 and (panel['x_bounds'][0] - 10 <= cx <= panel['x_bounds'][1] + 10):
            x_ticks.append((cx, float(word['text'])))
        if word['text'] in {'0.0', '0.5', '1.0', '1.5', '2.0'} and abs(cx - float(panel['y_tick_x'])) < 7 and (panel['y_bounds'][0] - 10 <= cy <= panel['y_bounds'][1] + 10):
            y_ticks.append((cy, float(word['text'])))
    tx_slope, tx_intercept, x_r2 = linear_fit(x_ticks)
    ey_slope, ey_intercept, y_r2 = linear_fit(y_ticks)
    return (tx_slope, tx_intercept, ey_slope, ey_intercept, x_r2, y_r2)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    calibrations = []
    with pdfplumber.open(PDF) as document:
        page = document.pages[5]
        for panel in PANELS:
            cal = calibrate(page, panel)
            calibrations.append({'figure': panel['figure'], 'time_r2': cal[4], 'expansion_r2': cal[5]})
            for color_name, position_mm in panel['positions'].items():
                color = COLORS[color_name]
                points = []
                for obj in page.curves:
                    if not obj.get('fill') or len(obj.get('pts') or []) != 5:
                        continue
                    if not (4.24 <= float(obj['x1'] - obj['x0']) <= 4.34 and 4.24 <= float(obj['bottom'] - obj['top']) <= 4.34):
                        continue
                    if obj.get('non_stroking_color') != color or obj.get('stroking_color') != color:
                        continue
                    cx = (obj['x0'] + obj['x1']) / 2
                    cy = (obj['top'] + obj['bottom']) / 2
                    if not (panel['x_bounds'][0] <= cx <= panel['x_bounds'][1] and panel['y_bounds'][0] <= cy <= panel['y_bounds'][1]):
                        continue
                    time_days = cal[0] * cx + cal[1]
                    expansion = cal[2] * cy + cal[3]
                    if time_days < 20 and expansion > 0.2:
                        continue
                    points.append((time_days, expansion, cx, cy))
                deduped = []
                for point in sorted(points):
                    if deduped and abs(point[0] - deduped[-1][0]) < 0.2 and (abs(point[1] - deduped[-1][1]) < 0.003):
                        continue
                    deduped.append(point)
                curve_id = f"Kawabata2023_{panel['figure']}_{position_mm}mm"
                for point_index, (time_days, expansion, source_x, source_y) in enumerate(deduped, start=1):
                    rows.append({'record_id': 'DEF-001010', 'source_short': 'Kawabata et al. 2023', 'DOI': '10.3151/jact.21.851', 'page_number': 6, 'figure_id': f"Kawabata2023_{panel['figure']}", 'curve_id': curve_id, 'point_index': point_index, 'time_days': round(time_days, 3), 'expansion_percent': round(expansion, 6), 'source_pdf_x': round(source_x, 3), 'source_pdf_top': round(source_y, 3), 'temperature_C': 90.0, 'treatment_duration_days': 0.5, 'storage_temperature_C': 15.25, 'water_cement_ratio': 0.49, 'aggregate_type': 'nonreactive_sand_gravel', 'moisture_condition': 'partial_tap_water_immersion', 'cement_type': 'high_early_strength_Portland', 'SO3_percent': 2.46, 'alkali_percent': 0.46, 'K2SO4_kg_m3': 15.24, 'specimen_type': 'prismatic_200x200mm', 'reinforcement_stirrups': int(bool(panel['stirrups'])), 'measurement_direction': panel['direction'], 'measurement_position_mm': position_mm, 'extraction_method': 'vector_marker_geometry', 'axis_calibration_r2_time': round(cal[4], 8), 'axis_calibration_r2_expansion': round(cal[5], 8), 'quality_status': 'accepted' if min(cal[4], cal[5]) >= 0.997 else 'review'})
    output = OUT / 'kawabata2023_vector_curves.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve_ids = sorted({str(row['curve_id']) for row in rows})
    summary = {'record_id': 'DEF-001010', 'curves': len(curve_ids), 'points': len(rows), 'curve_point_counts': {curve_id: sum((row['curve_id'] == curve_id for row in rows)) for curve_id in curve_ids}, 'calibrations': calibrations}
    (OUT / 'kawabata2023_extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
