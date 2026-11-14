from __future__ import annotations
import csv
import json
from pathlib import Path
import pdfplumber
ROOT = Path(__file__).resolve().parents[1]
PDF = next((ROOT / 'data/source_pdfs').glob('DEF-001001_*.pdf'))
OUT = ROOT / 'data/extracted'
MIXES = {'HR': {'cement_type': 'CEM_I_52.5_R', 'SO3_percent': 3.4, 'Al2O3_percent': 5.2, 'Na2Oeq_percent': 1.0, 'KOH_added_kg_m3': 1.745}, 'MR': {'cement_type': 'CEM_I_52.5_N', 'SO3_percent': 2.58, 'Al2O3_percent': 3.7, 'Na2Oeq_percent': 0.34, 'KOH_added_kg_m3': 0.0}}

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

def calibration(page: pdfplumber.page.Page, x_y: float, y_x: float, y_bounds: tuple[float, float], x_values: set[str], y_values: set[str]) -> tuple[float, float, float, float, float, float]:
    x_ticks = []
    y_ticks = []
    for word in page.extract_words():
        cx = (word['x0'] + word['x1']) / 2
        cy = (word['top'] + word['bottom']) / 2
        if word['text'] in x_values and abs(cy - x_y) < 7:
            x_ticks.append((cx, float(word['text'])))
        if word['text'] in y_values and abs(cx - y_x) < 8 and (y_bounds[0] <= cy <= y_bounds[1]):
            y_ticks.append((cy, float(word['text'])))
    tx_slope, tx_intercept, x_r2 = linear_fit(x_ticks)
    ey_slope, ey_intercept, y_r2 = linear_fit(y_ticks)
    return (tx_slope, tx_intercept, ey_slope, ey_intercept, x_r2, y_r2)

def marker_points(page: pdfplumber.page.Page, mix: str, size: float, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> list[tuple[float, float]]:
    points = []
    for obj in page.curves:
        if mix == 'HR':
            selected = obj.get('fill') and len(obj.get('pts') or []) == 5
        else:
            selected = not obj.get('fill') and obj.get('stroke') and (len(obj.get('pts') or []) == 6)
        if not selected or abs(float(obj['x1'] - obj['x0']) - size) > 0.045:
            continue
        cx = (obj['x0'] + obj['x1']) / 2
        cy = (obj['top'] + obj['bottom']) / 2
        if x_bounds[0] <= cx <= x_bounds[1] and y_bounds[0] <= cy <= y_bounds[1]:
            points.append((cx, cy))
    deduped = []
    for point in sorted(points):
        if deduped and abs(point[0] - deduped[-1][0]) < 0.08 and (abs(point[1] - deduped[-1][1]) < 0.08):
            continue
        deduped.append(point)
    return deduped

def add_rows(rows: list[dict[str, object]], page_number: int, figure_id: str, exposure: str, mix: str, points: list[tuple[float, float]], cal: tuple[float, float, float, float, float, float]) -> None:
    tx_slope, tx_intercept, ey_slope, ey_intercept, x_r2, y_r2 = cal
    chemistry = MIXES[mix]
    wet_days = ''
    dry_days = ''
    if exposure == 'wetting_drying':
        wet_days = 3 if mix == 'HR' else 4
        dry_days = 11 if mix == 'HR' else 10
    curve_id = f'Martin2023_{figure_id}_{mix}_{exposure}'
    for point_index, (source_x, source_y) in enumerate(points, start=1):
        rows.append({'record_id': 'DEF-001001', 'source_short': 'Martin et al. 2023', 'DOI': '10.1680/jmacr.22.00282', 'page_number': page_number, 'figure_id': f'Martin2023_{figure_id}', 'curve_id': curve_id, 'point_index': point_index, 'time_days': round(tx_slope * source_x + tx_intercept, 3), 'expansion_percent': round(ey_slope * source_y + ey_intercept, 6), 'source_pdf_x': round(source_x, 3), 'source_pdf_top': round(source_y, 3), 'temperature_C': 81.0, 'treatment_duration_days': 3.0, 'storage_temperature_C': 20.0, 'water_cement_ratio': 0.46, 'aggregate_type': 'siliceous_nonreactive', 'moisture_condition': exposure, 'wet_phase_days': wet_days, 'dry_phase_days': dry_days, 'dry_phase_RH_percent': 75 if exposure == 'wetting_drying' else '', **chemistry, 'SO3_Al2O3_ratio': round(chemistry['SO3_percent'] / chemistry['Al2O3_percent'], 5), 'extraction_method': 'vector_marker_geometry', 'axis_calibration_r2_time': round(x_r2, 8), 'axis_calibration_r2_expansion': round(y_r2, 8), 'quality_status': 'accepted' if min(x_r2, y_r2) >= 0.997 else 'review'})

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    calibrations = []
    with pdfplumber.open(PDF) as document:
        page6 = document.pages[5]
        cal4 = calibration(page6, 298.4, 69.2, (145, 270), {'0', '100', '200', '300', '400', '500'}, {'0', '0.5', '1.0', '1.5', '2.0', '2.5'})
        calibrations.append(('Fig4', *cal4[-2:]))
        for mix, size in (('HR', 3.58), ('MR', 3.62)):
            points = marker_points(page6, mix, size, (78.5, 278), (145, 270))
            add_rows(rows, 6, 'Fig4', 'continuous_immersion', mix, points, cal4)
        page7 = document.pages[6]
        cal6 = calibration(page7, 302.4, 73.2, (145, 280), {'0', '100', '200', '300', '400', '500'}, {'0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35'})
        calibrations.append(('Fig6', *cal6[-2:]))
        for mix, size in (('HR', 3.74), ('MR', 3.79)):
            points = marker_points(page7, mix, size, (80, 292), (145, 280))
            add_rows(rows, 7, 'Fig6', 'wetting_drying', mix, points, cal6)
    output = OUT / 'martin2023_vector_curves.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve_ids = sorted({str(row['curve_id']) for row in rows})
    summary = {'record_id': 'DEF-001001', 'curves': len(curve_ids), 'points': len(rows), 'curve_point_counts': {curve_id: sum((row['curve_id'] == curve_id for row in rows)) for curve_id in curve_ids}, 'calibrations': [{'figure': figure, 'time_r2': x_r2, 'expansion_r2': y_r2} for figure, x_r2, y_r2 in calibrations]}
    (OUT / 'martin2023_extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
