from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[1]
RENDERED = ROOT / 'results/corpus_analysis' / 'rendered_curve_candidates'
DATA = ROOT / 'data/extracted'
OUT = DATA / 'qa'
COLORS = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (230, 159, 0), (86, 180, 233)]
SPECS = [{'record': 'DEF-000170', 'csv': 'pavoine2007_vector_curves.csv', 'pages': {6: (35, 50, 505, 550)}}, {'record': 'DEF-000753', 'csv': 'tiecher2021_vector_curves.csv', 'pages': {6: (105, 55, 550, 315)}}, {'record': 'DEF-000998', 'csv': 'aubert2013_vector_curves.csv', 'pages': {3: (320, 125, 565, 285)}}, {'record': 'DEF-001001', 'csv': 'martin2023_vector_curves.csv', 'pages': {6: (55, 135, 290, 310), 7: (55, 135, 305, 315)}}, {'record': 'DEF-001010', 'csv': 'kawabata2023_vector_curves.csv', 'pages': {6: (65, 325, 535, 735)}}]

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        pdf_path = next((ROOT / 'data/source_pdfs').glob(f"{spec['record']}_*.pdf"))
        reader = PdfReader(str(pdf_path))
        with (DATA / spec['csv']).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        by_page: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_page[int(row['page_number'])].append(row)
        for page_number, crop_pdf in spec['pages'].items():
            source_path = RENDERED / f"{spec['record']}_page_{page_number:02d}.png"
            if not source_path.exists():
                continue
            image = Image.open(source_path).convert('RGB')
            page = reader.pages[page_number - 1]
            sx = image.width / float(page.mediabox.width)
            sy = image.height / float(page.mediabox.height)
            draw = ImageDraw.Draw(image)
            curve_ids = sorted({row['curve_id'] for row in by_page[page_number]})
            color_map = {curve_id: COLORS[index % len(COLORS)] for index, curve_id in enumerate(curve_ids)}
            for row in by_page[page_number]:
                x = float(row['source_pdf_x']) * sx
                y = float(row['source_pdf_top']) * sy
                radius = 3.5
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color_map[row['curve_id']], width=2)
            full_path = OUT / f"{spec['record']}_page{page_number:02d}_overlay.png"
            image.save(full_path, dpi=(300, 300))
            left, top, right, bottom = crop_pdf
            crop = image.crop((int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)))
            crop.save(OUT / f"{spec['record']}_page{page_number:02d}_plot_crop_overlay.png", dpi=(300, 300))
            print(f"{spec['record']} page {page_number}: points={len(by_page[page_number])}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
