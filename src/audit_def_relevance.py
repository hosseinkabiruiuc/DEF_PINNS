from __future__ import annotations
import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from pypdf import PdfReader
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA = PROJECT_ROOT / 'data/metadata' / 'master_metadata.csv'
VALID_DIR = PROJECT_ROOT / 'data/source_pdfs'
AUDIT_CSV = PROJECT_ROOT / 'data/screening' / 'strict_relevance_audit.csv'
AUDIT_JSON = PROJECT_ROOT / 'data/screening' / 'strict_relevance_audit_summary.json'
DIRECT_DEF = ('delayed ettringite formation', 'delayed ettringite', 'internal sulfate attack', 'internal sulphate attack')
ETTRINGITE = ('ettringite', 'aft phase', 'aft-phase', 'secondary aft')
CEMENTITIOUS = ('cement', 'cementitious', 'concrete', 'mortar', 'portland', 'clinker', 'cement paste')
EARLY_HEAT = ('heat curing', 'heat-cured', 'heat cured', 'steam curing', 'steam-cured', 'steam cured', 'thermal curing', 'high-temperature curing', 'high temperature curing', 'elevated curing temperature', 'early-age temperature', 'early age temperature', 'early thermal history', 'heat treatment')
DEF_BEHAVIOR = ('reformation', 're-formation', 'secondary ettringite', 'delayed expansion', 'expansion', 'swelling', 'length change', 'strain', 'decomposition', 'destabilization', 'instability', 'stability', 'moisture exposure', 'water exposure', 'water storage')
EXCLUDE_DOMAINS = ('dental', 'resin cement', 'bone cement', 'asphalt', 'soil cement', 'soil-cement', 'oil well cement', 'refractory concrete', 'polymer concrete', 'marl soil')

def normalize(value: str) -> str:
    return re.sub('\\s+', ' ', (value or '').lower()).strip()

def extract_text(path: Path, max_pages: int=15) -> str:
    reader = PdfReader(str(path), strict=False)
    chunks: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            chunks.append(page.extract_text() or '')
        except Exception:
            continue
    return normalize(' '.join(chunks))

def occurrences(text: str, terms: tuple[str, ...]) -> int:
    return sum((text.count(term) for term in terms))

def classify(title: str, text: str) -> tuple[str, str, int]:
    title = normalize(title)
    opening = text[:25000]
    combined = f'{title} {opening}'
    if any((term in title for term in EXCLUDE_DOMAINS)):
        return ('reject', 'out-of-scope material/application named in title', 0)
    title_direct = any((term in title for term in DIRECT_DEF))
    body_direct_count = occurrences(opening, DIRECT_DEF)
    has_cement = any((term in combined for term in CEMENTITIOUS))
    if title_direct and has_cement:
        return ('direct_DEF', 'title explicitly identifies DEF/internal sulfate attack', 100)
    if body_direct_count >= 2 and has_cement:
        return ('direct_DEF', 'DEF/internal sulfate attack discussed repeatedly in opening PDF text', 90)
    ettringite_count = occurrences(opening, ETTRINGITE)
    heat_count = occurrences(opening, EARLY_HEAT)
    behavior_count = occurrences(opening, DEF_BEHAVIOR)
    if has_cement and any((term in title for term in ETTRINGITE)) and any((term in title for term in EARLY_HEAT)) and any((term in title for term in DEF_BEHAVIOR)):
        return ('DEF_support', 'title links cementitious ettringite, early heat, and long-term behavior', 80)
    if has_cement and ettringite_count >= 3 and (heat_count >= 2) and (behavior_count >= 2):
        return ('DEF_support', 'opening text repeatedly links cementitious ettringite, early heat, and expansion/stability', 70)
    return ('reject', 'no sustained direct DEF or cementitious ettringite-plus-early-heat evidence', 0)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pages', type=int, default=15)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--pdf-dir', default=str(VALID_DIR))
    parser.add_argument('--output-prefix', default='strict_relevance_audit')
    args = parser.parse_args()
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = PROJECT_ROOT / pdf_dir
    audit_csv = PROJECT_ROOT / 'data/screening' / f'{args.output_prefix}.csv'
    audit_json = PROJECT_ROOT / 'data/screening' / f'{args.output_prefix}_summary.json'
    with METADATA.open(newline='', encoding='utf-8-sig') as handle:
        metadata = {row['record_id']: row for row in csv.DictReader(handle) if row.get('relevance_status') == 'retained'}
    raw_path = PROJECT_ROOT / 'data/metadata' / 'crossref_raw_records.json'
    if raw_path.exists():
        from collect_def_literature import item_pdf_priority, record_from_crossref
        raw_items = json.loads(raw_path.read_text(encoding='utf-8'))
        for index, item in enumerate(sorted(raw_items, key=item_pdf_priority, reverse=True), start=1):
            record = record_from_crossref(item, index)
            metadata.setdefault(record.record_id, {'title': record.title, 'DOI': record.doi})
    results: list[dict[str, str | int]] = []
    for index, path in enumerate(sorted(pdf_dir.glob('*.pdf')), start=1):
        match = re.search('DEF-\\d{6}', path.name)
        record_id = match.group(0) if match else path.stem
        row = metadata.get(record_id, {})
        try:
            text = extract_text(path, args.max_pages)
            status, reason, score = classify(row.get('title', ''), text)
        except Exception as exc:
            status, reason, score = ('reject', f'text extraction failed: {exc}', 0)
        results.append({'record_id': record_id, 'title': row.get('title', ''), 'DOI': row.get('DOI', ''), 'local_filename': str(path.relative_to(PROJECT_ROOT)), 'strict_relevance_status': status, 'strict_relevance_reason': reason, 'strict_relevance_score': score})
        if index % 50 == 0:
            print(f'Audited {index} PDFs', flush=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    counts = Counter((str(row['strict_relevance_status']) for row in results))
    summary = {'audited_pdfs': len(results), 'direct_DEF': counts['direct_DEF'], 'DEF_support': counts['DEF_support'], 'strict_retained_total': counts['direct_DEF'] + counts['DEF_support'], 'strict_rejects': counts['reject'], 'audit_csv': str(audit_csv.relative_to(PROJECT_ROOT)), 'applied': args.apply}
    if args.apply:
        quarantine = PROJECT_ROOT / 'data/rejected' / 'strict_relevance_reaudit'
        quarantine.mkdir(parents=True, exist_ok=True)
        moved = 0
        for row in results:
            if row['strict_relevance_status'] != 'reject':
                continue
            source = PROJECT_ROOT / str(row['local_filename'])
            if not source.exists():
                continue
            target = quarantine / source.name
            if target.exists():
                target = quarantine / f'reaudit_{source.name}'
            shutil.move(str(source), str(target))
            moved += 1
        summary['quarantined_pdfs'] = moved
    audit_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
