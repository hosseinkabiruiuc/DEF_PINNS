from __future__ import annotations
import csv
import json
import re
from pathlib import Path
import pdfplumber
ROOT = Path(__file__).resolve().parents[1]
PDF = next((ROOT / 'data/source_pdfs').glob('DEF-000998_*.pdf'))
OUT = ROOT / 'data/extracted'
SERIES = {'C20': {'kind': 'curve', 'pts': 4, 'width': 1.56, 'color': (0.24314, 0.23922, 0.24706), 'addition': 'none', 'addition_percent': 0.0, 'storage_C': 20.0}, 'CN20': {'kind': 'curve', 'pts': 9, 'width': 1.37, 'color': (0.24314, 0.23922, 0.24706), 'addition': 'NaOH', 'addition_percent': 1.75, 'storage_C': 20.0}, 'CS20': {'kind': 'rect', 'pts': 4, 'width': 1.56, 'color': (0.53333, 0.54118, 0.54902), 'addition': 'Na2SO4', 'addition_percent': 3.11, 'storage_C': 20.0}, 'CS40': {'kind': 'curve', 'pts': 3, 'width': 1.56, 'color': (0.48627, 0.49412, 0.50588), 'addition': 'Na2SO4', 'addition_percent': 3.11, 'storage_C': 40.0}}
BINDER_CHEMISTRY = {'C20': {'SiO2_percent': 19.3, 'Al2O3_percent': 4.6, 'Fe2O3_percent': 2.2, 'CaO_percent': 63.9, 'MgO_percent': 2.4, 'SO3_percent': 3.3, 'K2O_percent': 1.1, 'Na2O_percent': 0.3}, 'CN20': {'SiO2_percent': 18.9, 'Al2O3_percent': 4.5, 'Fe2O3_percent': 2.2, 'CaO_percent': 62.8, 'MgO_percent': 2.4, 'SO3_percent': 3.2, 'K2O_percent': 1.0, 'Na2O_percent': 1.6}, 'CS20': {'SiO2_percent': 18.7, 'Al2O3_percent': 4.4, 'Fe2O3_percent': 2.2, 'CaO_percent': 62.0, 'MgO_percent': 2.3, 'SO3_percent': 4.9, 'K2O_percent': 1.0, 'Na2O_percent': 1.6}, 'CS40': {'SiO2_percent': 18.7, 'Al2O3_percent': 4.4, 'Fe2O3_percent': 2.2, 'CaO_percent': 62.0, 'MgO_percent': 2.3, 'SO3_percent': 4.9, 'K2O_percent': 1.0, 'Na2O_percent': 1.6}}

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

def parsed_number(text: str) -> float | None:
    cleaned = re.sub('[^0-9-]', '.', text)
    cleaned = re.sub('\\.+', '.', cleaned).strip('.')
    try:
        return float(cleaned)
    except ValueError:
        return None

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(PDF) as document:
        page = document.pages[2]
        x_ticks = []
        y_ticks = []
        for word in page.extract_words():
            value = parsed_number(word['text'])
            if value is None:
                continue
            cx = (word['x0'] + word['x1']) / 2
            cy = (word['top'] + word['bottom']) / 2
            if 260 < cy < 275 and 340 < cx < 550 and (0 <= value <= 6):
                x_ticks.append((cx, value))
            if 325 < cx < 342 and 140 < cy < 265 and (0 <= value <= 1.4):
                y_ticks.append((cy, value))
        tx_slope, tx_intercept, x_r2 = linear_fit(x_ticks)
        ey_slope, ey_intercept, y_r2 = linear_fit(y_ticks)
        rows = []
        for series_id, spec in SERIES.items():
            objects = page.curves if spec['kind'] == 'curve' else page.rects
            points = []
            for obj in objects:
                if not obj.get('fill') or len(obj.get('pts') or []) != spec['pts']:
                    continue
                if abs(float(obj['x1'] - obj['x0']) - spec['width']) > 0.035:
                    continue
                if obj.get('non_stroking_color') != spec['color']:
                    continue
                cx = (obj['x0'] + obj['x1']) / 2
                cy = (obj['top'] + obj['bottom']) / 2
                if not (345 <= cx <= 547 and 147 <= cy <= 261):
                    continue
                age_years = tx_slope * cx + tx_intercept
                expansion = ey_slope * cy + ey_intercept
                points.append((age_years * 365.25, expansion, cx, cy))
            points.sort()
            curve_id = f'Aubert2013_Fig2_{series_id}'
            chemistry = BINDER_CHEMISTRY[series_id]
            for point_index, (time_days, expansion, source_x, source_y) in enumerate(points, start=1):
                rows.append({'record_id': 'DEF-000998', 'source_short': 'Aubert et al. 2013', 'DOI': '10.1680/adcr.11.00069', 'page_number': 3, 'figure_id': 'Aubert2013_Fig2', 'curve_id': curve_id, 'point_index': point_index, 'time_days': round(time_days, 3), 'expansion_percent': round(expansion, 6), 'source_pdf_x': round(source_x, 3), 'source_pdf_top': round(source_y, 3), 'temperature_C': 80.0, 'treatment_duration_days': 8.0 / 24.0, 'storage_temperature_C': spec['storage_C'], 'water_cement_ratio': 0.5, 'sand_cement_ratio': 3.0, 'aggregate_type': 'siliceous_standard_sand', 'moisture_condition': 'frequently_renewed_deionised_water', 'cement_type': 'CEM_I_52.5_R', 'blaine_cm2_g': 4200, 'chemical_addition': spec['addition'], 'chemical_addition_percent_cement': spec['addition_percent'], **chemistry, 'SO3_Al2O3_ratio': round(chemistry['SO3_percent'] / chemistry['Al2O3_percent'], 5), 'extraction_method': 'vector_marker_geometry', 'axis_calibration_r2_time': round(x_r2, 8), 'axis_calibration_r2_expansion': round(y_r2, 8), 'quality_status': 'accepted' if min(x_r2, y_r2) >= 0.997 else 'review'})
    output = OUT / 'aubert2013_vector_curves.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'record_id': 'DEF-000998', 'curves': len(SERIES), 'points': len(rows), 'curve_point_counts': {series_id: sum((row['curve_id'].endswith(series_id) for row in rows)) for series_id in SERIES}, 'axis_calibration_r2_time': x_r2, 'axis_calibration_r2_expansion': y_r2, 'reported_time_to_0_15_percent_days': {'CS40': 125, 'CS20': 180, 'CN20': 670, 'C20': 1460}}
    (OUT / 'aubert2013_extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
