from __future__ import annotations
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data/metadata' / 'master_metadata.csv'
PDF_DIR = ROOT / 'data/source_pdfs'
OUT = ROOT / 'results/corpus_analysis'
TEXT_DIR = OUT / 'fulltext'
TAXONOMY: dict[str, tuple[str, ...]] = {'expansion_kinetics': ('expansion', 'length change', 'swelling', 'strain', 'kinetic', 'induction period', 'dormant period', 'latency', 'sigmoid', 'time evolution'), 'thermal_history_threshold': ('heat curing', 'heat-cured', 'steam curing', 'steam-cured', 'curing temperature', 'thermal history', 'peak temperature', 'maximum temperature', 'temperature threshold'), 'moisture_exposure': ('relative humidity', 'moisture', 'water storage', 'water immersion', 'saturation', 'wetting', 'drying', 'humidity'), 'cement_chemistry': ('sulfate content', 'sulphate content', 'so3', 'alkali', 'c3a', 'aluminate', 'gypsum', 'clinker', 'pore solution', 'calcium sulfate'), 'scm_mitigation': ('fly ash', 'slag', 'silica fume', 'metakaolin', 'pozzolan', 'limestone', 'scm', 'supplementary cementitious', 'mitigation', 'inhibition'), 'aggregate_restraint_structure': ('aggregate', 'restraint', 'reinforcement', 'rebar', 'structural', 'bridge', 'field', 'anisotropy', 'stress', 'cracking'), 'microstructure_mechanism': ('microstructure', 'sem', 'scanning electron', 'xrd', 'x-ray diffraction', 'edx', 'crack', 'interface', 'c-s-h', 'monosulfate', 'aft', 'ettringite crystal'), 'modeling_numerical': ('model', 'numerical', 'simulation', 'finite element', 'constitutive', 'kinetic equation', 'differential equation', 'parameter identification', 'inverse analysis'), 'diagnostics_test_methods': ('test method', 'accelerated test', 'ultrasonic', 'resonance', 'petrographic', 'diagnosis', 'risk assessment', 'monitoring', 'measurement')}
RESPONSE_TERMS = ('expansion', 'length change', 'length-change', 'strain', 'swelling', 'deformation')
TIME_TERMS = ('time', 'age', 'day', 'days', 'month', 'months', 'year', 'years', 'week', 'weeks')
VISUAL_TERMS = ('figure', 'fig.', 'fig ', 'table', 'curve', 'plotted', 'versus', ' vs ')
DEF_TERMS = ('delayed ettringite', 'internal sulfate attack', 'internal sulphate attack', 'def-induced')
EQUATION_TERMS = ('kinetic model', 'rate equation', 'governing equation', 'differential equation', 'evolution law', 'arrhenius', 'activation energy', 'characteristic time', 'time constant', 'reaction degree', 'parameter identification', 'equation (', 'eq. (')
EMPIRICAL_TERMS = ('specimen', 'mortar bar', 'concrete prism', 'measured', 'experimental', 'mixture', 'mix design', 'storage', 'curing regime', 'length comparator')

def norm(value: str) -> str:
    return re.sub('\\s+', ' ', (value or '').lower()).strip()

def count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum((text.count(term) for term in terms))

def snippets(text: str, terms: tuple[str, ...], radius: int=240, limit: int=3) -> str:
    lowered = text.lower()
    hits: list[str] = []
    for term in terms:
        start = 0
        while len(hits) < limit:
            idx = lowered.find(term, start)
            if idx < 0:
                break
            left = max(0, idx - radius)
            right = min(len(text), idx + len(term) + radius)
            hits.append(re.sub('\\s+', ' ', text[left:right]).strip())
            start = idx + len(term)
        if len(hits) >= limit:
            break
    return ' || '.join(hits)

def classify(fulltext: str) -> tuple[str, str, dict[str, int]]:
    scores = {name: count_terms(fulltext, terms) for name, terms in TAXONOMY.items()}
    ordered = sorted(scores, key=lambda name: scores[name], reverse=True)
    primary = ordered[0] if ordered else 'uncategorized'
    secondary = ';'.join((name for name in ordered[1:4] if scores[name] > 0))
    return (primary, secondary, scores)

def page_curve_score(text: str) -> tuple[int, str]:
    response = count_terms(text, RESPONSE_TERMS)
    timing = count_terms(text, TIME_TERMS)
    visual = count_terms(text, VISUAL_TERMS)
    direct = count_terms(text, DEF_TERMS)
    empirical = count_terms(text, EMPIRICAL_TERMS)
    unit_cues = len(re.findall('(?:%|µm/m|microstrain|mm/m|mm|days?|months?|years?)', text, re.I))
    score = min(response, 8) * 4 + min(timing, 8) * 2 + min(visual, 6) * 3 + min(direct, 5) * 3
    score += min(empirical, 6) * 2 + min(unit_cues, 10)
    reasons = []
    if response:
        reasons.append(f'response={response}')
    if timing:
        reasons.append(f'time={timing}')
    if visual:
        reasons.append(f'figure_table={visual}')
    if direct:
        reasons.append(f'DEF={direct}')
    if empirical:
        reasons.append(f'empirical={empirical}')
    return (score, '; '.join(reasons))

def page_equation_score(text: str) -> tuple[int, str]:
    terms = count_terms(text, EQUATION_TERMS)
    math_cues = len(re.findall('(?:d\\s*[εe]|d\\s*alpha|exp\\s*\\(|ln\\s*\\(|k\\s*=|tau|\\u03c4|\\u03b1)', text, re.I))
    score = min(terms, 8) * 5 + min(math_cues, 12) * 2
    return (score, f'equation_terms={terms}; math_cues={math_cues}')

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with META.open(newline='', encoding='utf-8-sig') as handle:
        metadata = list(csv.DictReader(handle))
    paper_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    equation_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for paper_index, row in enumerate(metadata, start=1):
        pdf_path = ROOT / row['local_filename']
        try:
            reader = PdfReader(str(pdf_path), strict=False)
        except Exception as exc:
            errors.append({'record_id': row['record_id'], 'error': repr(exc)})
            continue
        pages: list[str] = []
        local_page_rows: list[dict[str, Any]] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ''
            except Exception as exc:
                raw_text = ''
                errors.append({'record_id': row['record_id'], 'error': f'page {page_index}: {exc!r}'})
            text = norm(raw_text)
            pages.append(raw_text)
            curve_score, curve_reason = page_curve_score(text)
            equation_score, equation_reason = page_equation_score(text)
            page_row = {'record_id': row['record_id'], 'title': row['title'], 'DOI': row['DOI'], 'local_filename': row['local_filename'], 'page_number': page_index, 'curve_score': curve_score, 'curve_reason': curve_reason, 'equation_score': equation_score, 'equation_reason': equation_reason, 'curve_snippet': snippets(raw_text, RESPONSE_TERMS), 'equation_snippet': snippets(raw_text, EQUATION_TERMS)}
            local_page_rows.append(page_row)
            if equation_score >= 10:
                equation_rows.append(page_row)
        fulltext_raw = '\n\n'.join(pages)
        fulltext = norm(fulltext_raw)
        (TEXT_DIR / f"{row['record_id']}.txt").write_text(fulltext_raw, encoding='utf-8')
        primary, secondary, scores = classify(fulltext)
        category_counts[primary] += 1
        top_curve_pages = sorted(local_page_rows, key=lambda item: int(item['curve_score']), reverse=True)[:5]
        top_equation_pages = sorted(local_page_rows, key=lambda item: int(item['equation_score']), reverse=True)[:3]
        likely_empirical = any((int(item['curve_score']) >= 35 for item in top_curve_pages))
        paper_rows.append({'record_id': row['record_id'], 'title': row['title'], 'authors': row['authors'], 'year': row['year'], 'journal': row['journal'], 'DOI': row['DOI'], 'local_filename': row['local_filename'], 'page_count': len(reader.pages), 'primary_category': primary, 'secondary_categories': secondary, 'category_scores_json': json.dumps(scores, sort_keys=True), 'direct_DEF_mentions': count_terms(fulltext, DEF_TERMS), 'expansion_mentions': count_terms(fulltext, RESPONSE_TERMS), 'time_mentions': count_terms(fulltext, TIME_TERMS), 'equation_mentions': count_terms(fulltext, EQUATION_TERMS), 'likely_empirical_curve_paper': likely_empirical, 'top_curve_pages': ';'.join((str(item['page_number']) for item in top_curve_pages)), 'top_curve_scores': ';'.join((str(item['curve_score']) for item in top_curve_pages)), 'top_equation_pages': ';'.join((str(item['page_number']) for item in top_equation_pages)), 'top_equation_scores': ';'.join((str(item['equation_score']) for item in top_equation_pages))})
        page_rows.extend((item for item in local_page_rows if int(item['curve_score']) >= 20))
        if paper_index % 10 == 0:
            print(f'Analyzed {paper_index}/{len(metadata)} papers', flush=True)
    paper_fields = list(paper_rows[0].keys())
    with (OUT / 'corpus_scientific_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=paper_fields)
        writer.writeheader()
        writer.writerows(paper_rows)
    page_fields = list(page_rows[0].keys())
    with (OUT / 'page_level_curve_candidates.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=page_fields)
        writer.writeheader()
        writer.writerows(sorted(page_rows, key=lambda item: int(item['curve_score']), reverse=True))
    with (OUT / 'governing_equation_evidence.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=page_fields)
        writer.writeheader()
        writer.writerows(sorted(equation_rows, key=lambda item: int(item['equation_score']), reverse=True))
    summary = {'papers_analyzed': len(paper_rows), 'pages_with_curve_score_at_least_20': len(page_rows), 'papers_likely_to_contain_empirical_curves': sum((str(row['likely_empirical_curve_paper']).lower() == 'true' for row in paper_rows)), 'pages_with_equation_score_at_least_10': len(equation_rows), 'primary_category_counts': dict(category_counts.most_common()), 'text_extraction_errors': len(errors)}
    (OUT / 'corpus_analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'text_extraction_errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
