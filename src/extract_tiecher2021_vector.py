from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import pdfplumber
ROOT = Path(__file__).resolve().parents[1]
PDF = next((ROOT / 'data/source_pdfs').glob('DEF-000753_*.pdf'))
OUT = ROOT / 'data/extracted'
MIXES = {(0.753, 0.0, 0.0): {'cement_type': 'CP V-ARI', 'scm_type': 'none', 'scm_percent': 0.0, 'CaO_percent': 0.93, 'SiO2_percent': 19.56, 'Al2O3_percent': 5.26, 'SO3_percent': 2.93, 'Na2Oeq_percent': 0.54, 'blaine_cm2_g': 4250}, (0.773, 0.353, 0.0667): {'cement_type': 'CP II-F', 'scm_type': 'limestone_filler', 'scm_percent': 6.0, 'CaO_percent': 0.74, 'SiO2_percent': 20.41, 'Al2O3_percent': 4.55, 'SO3_percent': 2.52, 'Na2Oeq_percent': 0.47, 'blaine_cm2_g': 4450}, (0.0, 0.502, 0.0): {'cement_type': 'CP II-Z', 'scm_type': 'fly_ash', 'scm_percent': 9.0, 'CaO_percent': 1.02, 'SiO2_percent': 22.69, 'Al2O3_percent': 5.82, 'SO3_percent': 1.85, 'Na2Oeq_percent': 0.41, 'blaine_cm2_g': 5110}, (0.0, 0.439, 0.753): {'cement_type': 'CP IV', 'scm_type': 'fly_ash', 'scm_percent': 24.0, 'CaO_percent': 0.85, 'SiO2_percent': 30.73, 'Al2O3_percent': 8.6, 'SO3_percent': 0.05, 'Na2Oeq_percent': 1.02, 'blaine_cm2_g': 4050}}

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

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(PDF) as document:
        page = document.pages[5]
        words = page.extract_words()
        x_ticks = []
        y_ticks = []
        for word in words:
            text = word['text']
            cx = (word['x0'] + word['x1']) / 2
            cy = (word['top'] + word['bottom']) / 2
            if text in {'0', '50', '100', '150', '200', '250', '300', '350', '400'} and 285 < cy < 305:
                x_ticks.append((cx, float(text)))
            if text in {'0.00', '0.20', '0.40', '0.60', '0.80', '1.00', '1.20', '1.40', '1.60', '1.80', '2.00'} and 110 < cx < 135:
                y_ticks.append((cy, float(text)))
        tx_slope, tx_intercept, x_r2 = linear_fit(x_ticks)
        ey_slope, ey_intercept, y_r2 = linear_fit(y_ticks)
        grouped: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for kind, objects in (('curve', page.curves), ('rect', page.rects)):
            for obj in objects:
                if not obj.get('fill'):
                    continue
                if abs(float(obj['x1'] - obj['x0']) - 4.0) > 0.15 or abs(float(obj['bottom'] - obj['top']) - 4.0) > 0.15:
                    continue
                color = obj.get('non_stroking_color')
                if not isinstance(color, tuple):
                    continue
                key = min(MIXES, key=lambda candidate: sum(((a - b) ** 2 for a, b in zip(candidate, color))))
                if sum(((a - b) ** 2 for a, b in zip(key, color))) > 0.002:
                    continue
                cx = (obj['x0'] + obj['x1']) / 2
                cy = (obj['top'] + obj['bottom']) / 2
                time_days = tx_slope * cx + tx_intercept
                expansion = ey_slope * cy + ey_intercept
                if not (-3 <= time_days <= 403 and -0.05 <= expansion <= 2.05):
                    continue
                if time_days < 140 and expansion > 1.5:
                    continue
                grouped[MIXES[key]['cement_type']].append((time_days, expansion, cx, cy))
    rows = []
    for cement_type, points in sorted(grouped.items()):
        mix = next((value for value in MIXES.values() if value['cement_type'] == cement_type))
        deduped = []
        for point in sorted(points):
            if deduped and abs(point[0] - deduped[-1][0]) < 0.2 and (abs(point[1] - deduped[-1][1]) < 0.002):
                continue
            deduped.append(point)
        curve_id = f"Tiecher2021_Fig2_{cement_type.replace(' ', '_')}"
        for point_index, (time_days, expansion, source_x, source_y) in enumerate(deduped, start=1):
            rows.append({'record_id': 'DEF-000753', 'source_short': 'Tiecher et al. 2021', 'DOI': '10.21041/ra.v11i3.537', 'page_number': 6, 'figure_id': 'Tiecher2021_Fig2', 'curve_id': curve_id, 'point_index': point_index, 'time_days': round(time_days, 3), 'expansion_percent': round(expansion, 6), 'source_pdf_x': round(source_x, 3), 'source_pdf_top': round(source_y, 3), 'temperature_C': 85.0, 'treatment_duration_days': 0.5, 'post_cure_exposure_C': 38.0, 'water_cement_ratio': 0.485, 'aggregate_type': 'ASR_innocuous_sand', 'moisture_condition': 'continuous_water_immersion', **mix, 'SO3_Al2O3_ratio': round(mix['SO3_percent'] / mix['Al2O3_percent'], 5), 'extraction_method': 'vector_marker_geometry', 'axis_calibration_r2_time': round(x_r2, 8), 'axis_calibration_r2_expansion': round(y_r2, 8), 'quality_status': 'accepted' if min(x_r2, y_r2) >= 0.997 else 'review'})
    output = OUT / 'tiecher2021_vector_curves.csv'
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'record_id': 'DEF-000753', 'curves': len(grouped), 'points': len(rows), 'curve_point_counts': {curve: sum((row['cement_type'] == curve for row in rows)) for curve in grouped}, 'axis_calibration_r2_time': x_r2, 'axis_calibration_r2_expansion': y_r2, 'cross_check_terminal_expansion_percent': {'CP V-ARI': 1.79, 'CP II-F': 0.45, 'CP II-Z': 0.08, 'CP IV': 0.1}}
    (OUT / 'tiecher2021_extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
