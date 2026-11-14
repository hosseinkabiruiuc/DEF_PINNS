from __future__ import annotations
import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any
import collect_def_literature as base
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / 'data/metadata'
DEEP_QUERIES = [('deep_direct_DEF', '"delayed ettringite formation"'), ('deep_direct_DEF', '"delayed ettringite" concrete'), ('deep_direct_DEF', '"delayed ettringite" mortar'), ('deep_direct_DEF', '"delayed ettringite" cement paste'), ('deep_direct_DEF', '"delayed ettringite" expansion'), ('deep_direct_DEF', '"delayed ettringite" moisture'), ('deep_direct_DEF', '"delayed ettringite" heat curing'), ('deep_direct_DEF', '"delayed ettringite" steam curing'), ('deep_internal_sulfate', '"internal sulfate attack" ettringite concrete'), ('deep_internal_sulfate', '"internal sulphate attack" ettringite concrete'), ('deep_internal_sulfate', '"internal sulfate attack" heat cured concrete'), ('deep_internal_sulfate', '"internal sulphate attack" heat cured concrete'), ('deep_heat_ettringite', '"heat cured" cement ettringite expansion'), ('deep_heat_ettringite', '"steam cured" cement ettringite expansion'), ('deep_heat_ettringite', '"high temperature curing" cement ettringite'), ('deep_heat_ettringite', '"elevated curing temperature" ettringite mortar'), ('deep_heat_ettringite', '"early age temperature" concrete ettringite expansion'), ('deep_heat_ettringite', '"ettringite reformation" cement heat'), ('deep_heat_ettringite', '"secondary ettringite" steam cured concrete'), ('deep_heat_ettringite', '"ettringite decomposition" cement heat curing'), ('deep_heat_ettringite', '"ettringite stability" cement curing temperature'), ('deep_heat_ettringite', '"delayed expansion" heat cured concrete ettringite')]

def key_for_item(item: dict[str, Any]) -> str:
    doi = base.clean_text(item.get('DOI')).lower()
    title = base.clean_text(item.get('title'))
    return doi or base.normalize_key(title)

def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))

def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-valid', type=int, default=600)
    parser.add_argument('--rows-per-query', type=int, default=500)
    parser.add_argument('--use-openalex-discovery', action='store_true')
    parser.add_argument('--use-semantic-scholar', action='store_true')
    args = parser.parse_args()
    base.ensure_dirs()
    metadata_path = METADATA_DIR / 'master_metadata.csv'
    existing_rows = load_rows(metadata_path)
    retained_rows = [row for row in existing_rows if row.get('relevance_status') == 'retained']
    known_items = json.loads((METADATA_DIR / 'crossref_raw_records.json').read_text(encoding='utf-8'))
    known_keys = {key_for_item(item) for item in known_items if key_for_item(item)}
    known_keys.update((row.get('DOI', '').lower() or base.normalize_key(row.get('title', '')) for row in existing_rows))
    max_numeric_id = 0
    for row in existing_rows:
        try:
            max_numeric_id = max(max_numeric_id, int(row.get('record_id', '').split('-')[-1]))
        except ValueError:
            continue
    print('Running deep Crossref discovery', flush=True)
    discovered = base.crossref_discover(DEEP_QUERIES, args.rows_per_query, 20000)
    if args.use_openalex_discovery:
        print('Running deep OpenAlex OA discovery', flush=True)
        discovered.extend(base.openalex_discover(DEEP_QUERIES, min(args.rows_per_query, 200), 10000, known_keys))
    new_items: list[dict[str, Any]] = []
    seen_new: set[str] = set()
    for item in discovered:
        key = key_for_item(item)
        if not key or key in known_keys or key in seen_new:
            continue
        seen_new.add(key)
        new_items.append(item)
    (METADATA_DIR / 'deep_discovery_new_records.json').write_text(json.dumps(new_items, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Found {len(new_items)} new metadata records', flush=True)
    hashes = {row.get('SHA256', '') for row in retained_rows if row.get('SHA256')}
    dois = {row.get('DOI', '').lower() for row in retained_rows if row.get('DOI')}
    titles = {base.normalize_key(row.get('title', '')) for row in retained_rows if row.get('title')}
    additions: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    attempted = 0
    for item in sorted(new_items, key=base.item_pdf_priority, reverse=True):
        max_numeric_id += 1
        record = base.record_from_crossref(item, max_numeric_id)
        doi_key = record.doi.lower()
        title_key = base.normalize_key(record.title)
        if doi_key and doi_key in dois or (title_key and title_key in titles):
            continue
        record.duplicate_status = 'unique_metadata'
        status, reason = base.metadata_relevance(record)
        record.relevance_status = status
        record.relevance_reason = reason
        if status != 'metadata_relevant':
            continue
        urls, skipped = base.candidate_pdf_urls(item, record.doi, use_s2=args.use_semantic_scholar, use_openalex=True, use_unpaywall=False, unpaywall_email='')
        for skipped_url in skipped:
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'pdf_discovery', 'reason': skipped_url['reason'], 'url': skipped_url['url'], 'query_family': record.query_family})
        for url in urls:
            attempted += 1
            raw_path = base.DIRS['raw'] / base.safe_filename(record)
            try:
                data, _ = base.request_bytes(url)
                if len(data) < 10000 or base.looks_like_html(data):
                    raise ValueError('download is too small or appears to be HTML')
                raw_path.write_bytes(data)
                valid, validation_reason, page_count = base.validate_pdf(raw_path)
                if not valid:
                    raise ValueError(validation_reason)
                digest = base.sha256_file(raw_path)
                if digest in hashes:
                    raise ValueError('duplicate SHA-256')
                text = base.extract_pdf_text(raw_path, max_pages=15)
                pdf_status, pdf_reason = base.pdf_relevance(record, text)
                if pdf_status != 'retained':
                    raise ValueError(pdf_reason)
            except Exception as exc:
                if raw_path.exists():
                    target = base.DIRS['rejected'] / raw_path.name
                    if target.exists():
                        target = base.DIRS['rejected'] / f'deep_{raw_path.name}'
                    shutil.move(str(raw_path), str(target))
                rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'deep_download_or_screening', 'reason': repr(exc), 'url': url, 'query_family': record.query_family})
                continue
            valid_path = base.DIRS['valid'] / raw_path.name
            shutil.move(str(raw_path), str(valid_path))
            record.pdf_source = url
            record.local_filename = str(valid_path.relative_to(PROJECT_ROOT))
            record.sha256 = digest
            record.page_count = str(page_count)
            record.pdf_valid = True
            record.duplicate_status = 'unique_pdf'
            record.relevance_status = 'retained'
            record.relevance_reason = pdf_reason
            record.candidate_curve, record.curve_candidate_reason = base.curve_candidate(record, text)
            additions.append(record.row())
            hashes.add(digest)
            if doi_key:
                dois.add(doi_key)
            if title_key:
                titles.add(title_key)
            print(f'Strict corpus: {len(retained_rows) + len(additions)} PDFs', flush=True)
            break
        if len(retained_rows) + len(additions) >= args.target_valid:
            break
    master_fields = list(base.Record('x').row().keys())
    combined_retained = retained_rows + additions
    write_rows(metadata_path, combined_retained, master_fields)
    (METADATA_DIR / 'master_metadata.json').write_text(json.dumps(combined_retained, indent=2, ensure_ascii=False), encoding='utf-8')
    write_rows(base.DIRS['screening'] / 'retained_screening_summary.csv', combined_retained, master_fields)
    candidates = [row for row in combined_retained if str(row.get('candidate_curve', '')).lower() == 'true']
    write_rows(base.DIRS['candidates'] / 'curve_candidate_manifest.csv', candidates, master_fields)
    prior_rejects = load_rows(METADATA_DIR / 'rejection_log.csv')
    reject_fields = ['record_id', 'title', 'doi', 'stage', 'reason', 'url', 'query_family']
    write_rows(METADATA_DIR / 'rejection_log.csv', prior_rejects + rejects, reject_fields)
    (METADATA_DIR / 'rejection_log.json').write_text(json.dumps(prior_rejects + rejects, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {'deep_queries': len(DEEP_QUERIES), 'new_metadata_records': len(new_items), 'deep_pdf_attempts': attempted, 'strict_pdfs_before_deep_search': len(retained_rows), 'strict_pdfs_added': len(additions), 'final_strict_retained_pdfs': len(combined_retained), 'likely_time_dependent_curve_candidates': len(candidates), 'target_valid_pdfs': args.target_valid, 'target_reached': len(combined_retained) >= args.target_valid}
    (METADATA_DIR / 'summary_counts.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (METADATA_DIR / 'deep_extension_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0 if summary['target_reached'] else 2
if __name__ == '__main__':
    raise SystemExit(main())
