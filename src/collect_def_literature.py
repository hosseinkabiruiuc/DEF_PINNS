from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
try:
    from pypdf import PdfReader
except Exception as exc:
    raise SystemExit('Missing dependency: pypdf is required for PDF validation.') from exc
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIRS = {'metadata': PROJECT_ROOT / 'data/metadata', 'raw': PROJECT_ROOT / 'data/downloads_raw', 'valid': PROJECT_ROOT / 'data/source_pdfs', 'rejected': PROJECT_ROOT / 'data/rejected', 'screening': PROJECT_ROOT / 'data/screening', 'candidates': PROJECT_ROOT / 'results/curve_candidates', 'logs': PROJECT_ROOT / 'logs'}
USER_AGENT = 'DEF-PINNs-literature-workflow/0.1 (mailto:research-workflow@example.com; legal OA scholarly discovery)'
SKIP_DOWNLOAD_DOMAINS = {'mdpi.com': 'known unavailable in this environment from prior ASR workflow', 'www.mdpi.com': 'known unavailable in this environment from prior ASR workflow', 'doi.org': 'DOI landing page is not a direct PDF', 'dx.doi.org': 'DOI landing page is not a direct PDF', 'api.elsevier.com': 'Elsevier API endpoint does not provide accessible PDF without credentials', 'www.sciencedirect.com': 'ScienceDirect PDF endpoint returned forbidden in this environment', 'sciencedirect.com': 'ScienceDirect PDF endpoint returned forbidden in this environment'}
QUERY_FAMILIES: dict[str, list[str]] = {'core_def': ['"delayed ettringite formation"', '"DEF" concrete', '"DEF" cement', '"delayed ettringite expansion"', '"internal sulfate attack" "delayed ettringite"'], 'heat_curing': ['"secondary ettringite" "heat treatment"', '"heat-cured concrete" expansion ettringite', '"heat cured concrete" expansion ettringite', '"high-temperature curing" ettringite', '"heat treatment" "cementitious materials" ettringite', '"thermal history" concrete DEF', '"curing temperature" DEF', '"sulfate-related expansion" "heat curing"'], 'kinetics_expansion': ['"delayed ettringite formation" expansion time', '"delayed ettringite formation" strain time', '"delayed ettringite formation" "length change"', '"delayed ettringite formation" kinetics', '"ettringite reformation" expansion', '"DEF" "length change" concrete', '"DEF" expansion kinetics cement'], 'materials_exposure': ['"delayed ettringite formation" "cement chemistry"', '"delayed ettringite formation" "supplementary cementitious materials"', '"delayed ettringite formation" aggregate', '"delayed ettringite formation" moisture', '"moisture exposure" DEF concrete', '"delayed ettringite formation" exposure concrete', '"delayed ettringite formation" microstructure', '"delayed ettringite formation" mortar', '"delayed ettringite formation" paste'], 'broader_def_context': ['ettringite "delayed formation" cement', 'ettringite "delayed formation" concrete', 'ettringite "delayed formation" mortar', 'ettringite "delayed formation" paste', '"internal sulfate attack" concrete heat curing', '"internal sulphate attack" concrete heat curing'], 'thermal_aft_context': ['"AFt" "delayed ettringite" concrete', '"AFt" "heat curing" cement', '"ettringite stability" cement temperature', '"ettringite decomposition" cement heat', '"ettringite reformation" "cement paste"', '"steam curing" concrete ettringite', '"steam-cured" concrete ettringite', '"precast concrete" "delayed ettringite formation"', '"mass concrete" "delayed ettringite formation"', '"high early temperature" concrete ettringite', '"elevated curing temperature" mortar expansion', '"heat induced" ettringite concrete expansion'], 'thermal_curing_expansion': ['"steam curing" concrete expansion', '"steam curing" mortar expansion', '"steam curing" cement paste expansion', '"heat curing" concrete expansion', '"heat curing" mortar expansion', '"heat curing" cement paste expansion', '"high temperature curing" concrete expansion', '"high temperature curing" mortar expansion', '"elevated temperature curing" concrete expansion', '"elevated temperature curing" mortar expansion', '"thermal curing" cementitious expansion', '"early age temperature" concrete expansion', '"early-age temperature" concrete expansion', '"delayed expansion" heat cured concrete', '"delayed expansion" steam cured concrete', '"delayed expansion" cement paste ettringite'], 'sulfate_ettringite_expansion': ['"sulfate attack" heat cured concrete', '"sulphate attack" heat cured concrete', '"sulfate attack" steam cured concrete', '"sulfate attack" concrete ettringite expansion', '"sulfate attack" mortar ettringite expansion', '"internal sulfate" heat cured concrete', '"internal sulphate" heat cured concrete', '"internal sulfate" ettringite expansion', '"internal sulphate" ettringite expansion', '"sulfate resistance" heat cured concrete ettringite', '"sulphate resistance" heat cured concrete ettringite'], 'phase_chemistry_def_support': ['"ettringite stability" cement hydration temperature', '"ettringite decomposition" cement hydration temperature', '"ettringite reformation" sulfate cement paste', '"AFt phase" cement temperature curing', '"monosulfate" ettringite reformation cement', '"aluminate" sulfate ettringite heat curing cement', '"calcium sulfoaluminate" ettringite heat curing', '"gypsum" ettringite heat curing cement', '"limestone filler" delayed ettringite formation', '"fly ash" delayed ettringite formation', '"slag" delayed ettringite formation', '"metakaolin" delayed ettringite formation']}
COMBO_TERMS = ['expansion', 'strain', '"length change"', 'kinetics', 'time', 'temperature', 'curing', '"cement chemistry"', '"supplementary cementitious materials"', 'aggregate', 'moisture', 'exposure', 'microstructure', 'concrete', 'mortar', 'paste']
CORE_TERMS = ['delayed ettringite formation', 'delayed ettringite', 'ettringite reformation', 'secondary ettringite', 'aft', 'internal sulfate attack', 'internal sulphate attack', 'sulfate-related expansion', 'sulphate-related expansion', 'sulfate attack', 'sulphate attack', 'delayed expansion']
DIRECT_DEF_TERMS = ['delayed ettringite formation', 'delayed ettringite', 'internal sulfate attack', 'internal sulphate attack']
ETTRINGITE_TERMS = ['ettringite', 'aft phase', 'aft-phase', 'secondary aft']
EARLY_HEAT_TERMS = ['heat curing', 'heat-cured', 'heat cured', 'steam curing', 'steam-cured', 'steam cured', 'thermal curing', 'high-temperature curing', 'high temperature curing', 'elevated curing temperature', 'early-age temperature', 'early age temperature', 'early thermal history', 'heat treatment']
DEF_BEHAVIOR_TERMS = ['reformation', 're-formation', 'secondary ettringite', 'delayed expansion', 'expansion', 'swelling', 'length change', 'strain', 'decomposition', 'stability', 'moisture exposure', 'water exposure', 'water storage']
OUT_OF_SCOPE_TERMS = ['dental', 'resin cement', 'bone cement', 'asphalt', 'soil cement', 'soil-cement', 'oil well cement', 'refractory concrete', 'polymer concrete']
CEMENT_TERMS = ['cement', 'cementitious', 'concrete', 'mortar', 'paste', 'binder', 'portland', 'clinker']
DEF_CONTEXT_TERMS = ['heat', 'thermal', 'temperature', 'curing', 'cured', 'moisture', 'water', 'expansion', 'strain', 'length change', 'durability', 'steam']
CURVE_TERMS = ['expansion', 'strain', 'length change', 'length-change', 'deformation', 'swelling']
TIME_TERMS = ['time', 'age', 'day', 'days', 'month', 'months', 'year', 'years', 'kinetics']
HIGH_YIELD_DOMAINS = ['link.springer.com', 'www.scientific.net', 'scientific.net', 'www.jstage.jst.go.jp', 'jstage.jst.go.jp', 'dl.astm.org', 'www.icevirtuallibrary.com', 'icevirtuallibrary.com', 'downloads.hindawi.com', 'www.scielo.br', 'scielo.br', 'revistaalconpat.org', 'materconstrucc.revistas.csic.es', 'iopscience.iop.org', 'stacks.iop.org', 'challengejournal.com', 'www.challengejournal.com', 'onlinelibrary.wiley.com', 'ceramics.onlinelibrary.wiley.com', 'journals.sagepub.com', 'ascelibrary.org']

@dataclass
class Record:
    record_id: str
    title: str = ''
    authors: str = ''
    year: str = ''
    journal: str = ''
    doi: str = ''
    query_family: str = ''
    discovery_source: str = ''
    pdf_source: str = ''
    local_filename: str = ''
    sha256: str = ''
    page_count: str = ''
    pdf_valid: bool = False
    duplicate_status: str = ''
    relevance_status: str = ''
    relevance_reason: str = ''
    notes: str = ''
    abstract: str = ''
    keywords: str = ''
    candidate_curve: bool = False
    curve_candidate_reason: str = ''
    attempted_urls: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {'record_id': self.record_id, 'title': self.title, 'authors': self.authors, 'year': self.year, 'journal': self.journal, 'DOI': self.doi, 'query_family': self.query_family, 'discovery_source': self.discovery_source, 'PDF_source': self.pdf_source, 'local_filename': self.local_filename, 'SHA256': self.sha256, 'page_count': self.page_count, 'pdf_valid': self.pdf_valid, 'duplicate_status': self.duplicate_status, 'relevance_status': self.relevance_status, 'relevance_reason': self.relevance_reason, 'notes': self.notes, 'candidate_curve': self.candidate_curve, 'curve_candidate_reason': self.curve_candidate_reason}

def ensure_dirs() -> None:
    for directory in DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)

def clean_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        value = ' '.join((str(v) for v in value))
    value = re.sub('<[^>]+>', ' ', str(value))
    value = unicodedata.normalize('NFKC', value)
    return re.sub('\\s+', ' ', value).strip()

def normalize_key(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub('[^a-z0-9]+', ' ', value)
    return re.sub('\\s+', ' ', value).strip()

def request_json(url: str, sleep_s: float=0.25, timeout: int=20) -> dict[str, Any]:
    time.sleep(sleep_s)
    req = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        data = response.read()
    return json.loads(data.decode('utf-8', errors='replace'))

def request_bytes(url: str, timeout: int=10, max_bytes: int=80000000) -> tuple[bytes, str]:
    req = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/pdf,application/octet-stream;q=0.9,*/*;q=0.2'})
    with urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get('content-type', '')
        chunks = []
        total = 0
        while True:
            chunk = response.read(512 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError('download exceeded max_bytes')
    return (b''.join(chunks), content_type)

def build_queries(limit: int | None=None) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for family, base_queries in QUERY_FAMILIES.items():
        for query in base_queries:
            queries.append((family, query))
    for core in ['"delayed ettringite formation"', '"internal sulfate attack" concrete', '"secondary ettringite" concrete', '"ettringite reformation" cement']:
        for term in COMBO_TERMS:
            queries.append(('systematic_combinations', f'{core} {term}'))
    seen = set()
    unique = []
    for family, query in queries:
        key = query.lower()
        if key not in seen:
            seen.add(key)
            unique.append((family, query))
    return unique[:limit] if limit else unique

def crossref_discover(queries: list[tuple[str, str]], rows_per_query: int, max_records: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()
    for family, query in queries:
        if len(records) >= max_records:
            break
        print(f'Crossref query [{family}]: {query}', flush=True)
        cursor = '*'
        fetched = 0
        while fetched < rows_per_query and len(records) < max_records:
            rows = min(100, rows_per_query - fetched)
            url = f'https://api.crossref.org/works?query={quote(query)}&filter=type:journal-article&rows={rows}&cursor={quote(cursor)}&select=DOI,title,author,published-print,published-online,container-title,abstract,link,URL,subject'
            try:
                payload = request_json(url)
            except Exception as exc:
                log_event('discovery_errors.jsonl', {'source': 'crossref', 'query': query, 'error': repr(exc)})
                break
            message = payload.get('message', {})
            items = message.get('items', [])
            cursor = message.get('next-cursor', cursor)
            if not items:
                break
            for item in items:
                doi = clean_text(item.get('DOI')).lower()
                title = clean_text(item.get('title'))
                key = doi or normalize_key(title)
                if not key or key in seen:
                    continue
                seen.add(key)
                item['_query_family'] = family
                item['_query'] = query
                item['_discovery_source'] = 'Crossref'
                records.append(item)
                if len(records) >= max_records:
                    break
            fetched += len(items)
            if len(items) < rows:
                break
    return records

def openalex_discover(queries: list[tuple[str, str]], rows_per_query: int, max_records: int, seen_existing: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set(seen_existing)
    for family, query in queries:
        if len(records) >= max_records:
            break
        print(f'OpenAlex query [{family}]: {query}', flush=True)
        page = 1
        fetched = 0
        while fetched < rows_per_query and len(records) < max_records:
            per_page = min(100, rows_per_query - fetched)
            url = f'https://api.openalex.org/works?search={quote(query)}&filter=type:article,is_oa:true&per-page={per_page}&page={page}&select=doi,title,authorships,publication_year,primary_location,best_oa_location,locations,abstract_inverted_index,concepts'
            try:
                payload = request_json(url)
            except Exception as exc:
                log_event('discovery_errors.jsonl', {'source': 'openalex_discovery', 'query': query, 'error': repr(exc)})
                break
            results = payload.get('results') or []
            if not results:
                break
            for work in results:
                doi_url = clean_text(work.get('doi'))
                doi = doi_url.replace('https://doi.org/', '').lower()
                title = clean_text(work.get('title'))
                key = doi or normalize_key(title)
                if not key or key in seen:
                    continue
                seen.add(key)
                item = openalex_to_crossref_like(work, family, query)
                records.append(item)
                if len(records) >= max_records:
                    break
            fetched += len(results)
            page += 1
            if len(results) < per_page:
                break
    return records

def abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ''
    positions: list[tuple[int, str]] = []
    for word, offsets in index.items():
        if not isinstance(offsets, list):
            continue
        for offset in offsets:
            if isinstance(offset, int):
                positions.append((offset, word))
    return clean_text(' '.join((word for _, word in sorted(positions))))

def openalex_to_crossref_like(work: dict[str, Any], family: str, query: str) -> dict[str, Any]:
    doi_url = clean_text(work.get('doi'))
    doi = doi_url.replace('https://doi.org/', '').lower()
    authors = []
    for authorship in work.get('authorships') or []:
        author = authorship.get('author') or {}
        name = clean_text(author.get('display_name'))
        if name:
            parts = name.split()
            authors.append({'given': ' '.join(parts[:-1]), 'family': parts[-1] if parts else ''})
    host = work.get('host_venue') or {}
    if not host:
        primary = work.get('primary_location') or {}
        source = primary.get('source') or {}
        host = {'display_name': source.get('display_name')}
    links = []
    locations = [work.get('best_oa_location') or {}, work.get('primary_location') or {}] + (work.get('locations') or [])
    seen_links = set()
    for location in locations:
        if not isinstance(location, dict):
            continue
        for key in ('pdf_url', 'landing_page_url'):
            url = clean_text(location.get(key))
            if url and url not in seen_links:
                seen_links.add(url)
                links.append({'URL': url, 'content-type': 'application/pdf' if key == 'pdf_url' else '', 'intended-application': 'open-access-full-text'})
    concepts = '; '.join((clean_text(c.get('display_name')) for c in (work.get('concepts') or [])[:12]))
    return {'DOI': doi, 'title': [clean_text(work.get('title'))], 'author': authors, 'published-online': {'date-parts': [[work.get('publication_year')]]} if work.get('publication_year') else {}, 'container-title': [clean_text(host.get('display_name'))], 'abstract': abstract_from_inverted_index(work.get('abstract_inverted_index')), 'subject': [concepts] if concepts else [], 'link': links, 'URL': doi_url, '_query_family': family, '_query': query, '_discovery_source': 'OpenAlex'}

def semantic_scholar_pdf_url(doi: str) -> str:
    if not doi:
        return ''
    url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}?fields=openAccessPdf,url,title'
    try:
        payload = request_json(url, sleep_s=1.0)
    except Exception as exc:
        log_event('discovery_errors.jsonl', {'source': 'semantic_scholar', 'doi': doi, 'error': repr(exc)})
        return ''
    pdf = payload.get('openAccessPdf') or {}
    return clean_text(pdf.get('url'))

def openalex_pdf_url(doi: str) -> str:
    if not doi:
        return ''
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    try:
        payload = request_json(url, sleep_s=0.25)
    except Exception as exc:
        log_event('discovery_errors.jsonl', {'source': 'openalex', 'doi': doi, 'error': repr(exc)})
        return ''
    best = payload.get('best_oa_location') or {}
    primary = payload.get('primary_location') or {}
    locations = [best, primary] + (payload.get('locations') or [])
    for location in locations:
        if not isinstance(location, dict):
            continue
        for key in ('pdf_url', 'landing_page_url'):
            value = clean_text(location.get(key))
            if value and ('pdf' in value.lower() or key == 'pdf_url'):
                return value
    return ''

def unpaywall_pdf_url(doi: str, email: str) -> str:
    if not doi or not email:
        return ''
    url = f'https://api.unpaywall.org/v2/{quote(doi)}?email={quote(email)}'
    try:
        payload = request_json(url, sleep_s=0.25)
    except Exception as exc:
        log_event('discovery_errors.jsonl', {'source': 'unpaywall', 'doi': doi, 'error': repr(exc)})
        return ''
    best = payload.get('best_oa_location') or {}
    for key in ('url_for_pdf', 'url'):
        value = clean_text(best.get(key))
        if value:
            return value
    for location in payload.get('oa_locations') or []:
        value = clean_text(location.get('url_for_pdf') or location.get('url'))
        if value:
            return value
    return ''

def record_from_crossref(item: dict[str, Any], index: int) -> Record:
    title = clean_text(item.get('title'))
    authors = []
    for author in item.get('author') or []:
        name = clean_text(' '.join(filter(None, [author.get('given'), author.get('family')])))
        if name:
            authors.append(name)
    year = ''
    for key in ('published-print', 'published-online'):
        parts = ((item.get(key) or {}).get('date-parts') or [[]])[0]
        if parts:
            year = str(parts[0])
            break
    journal = clean_text(item.get('container-title'))
    keywords = '; '.join((clean_text(s) for s in item.get('subject') or []))
    return Record(record_id=f'DEF-{index:06d}', title=title, authors='; '.join(authors), year=year, journal=journal, doi=clean_text(item.get('DOI')).lower(), query_family=clean_text(item.get('_query_family')), discovery_source=clean_text(item.get('_discovery_source')) or 'Crossref', abstract=clean_text(item.get('abstract')), keywords=keywords, notes=f"query={clean_text(item.get('_query'))}")

def metadata_relevance(record: Record) -> tuple[str, str]:
    haystack = normalize_key(' '.join([record.title, record.abstract, record.keywords, record.journal]))
    has_cement = any((term in haystack for term in CEMENT_TERMS))
    if any((term in normalize_key(record.title) for term in OUT_OF_SCOPE_TERMS)):
        return ('metadata_reject', 'out-of-scope material/application named in title')
    if has_cement and any((term in haystack for term in DIRECT_DEF_TERMS)):
        return ('metadata_relevant', 'title/abstract/keywords explicitly identify DEF or internal sulfate attack')
    has_ettringite = any((term in haystack for term in ETTRINGITE_TERMS))
    has_early_heat = any((term in haystack for term in EARLY_HEAT_TERMS))
    has_behavior = any((term in haystack for term in DEF_BEHAVIOR_TERMS))
    if has_cement and has_ettringite and has_early_heat and has_behavior:
        return ('metadata_relevant', 'ettringite in cementitious material linked to early heat and DEF-type behavior')
    return ('metadata_reject', 'title/abstract/keywords lack direct DEF or ettringite-plus-early-heat evidence')

def candidate_pdf_urls(item: dict[str, Any], doi: str, use_s2: bool, use_openalex: bool, use_unpaywall: bool, unpaywall_email: str) -> tuple[list[str], list[dict[str, str]]]:
    urls: list[str] = []
    skipped: list[dict[str, str]] = []
    urls.extend(doi_pdf_candidates(doi))
    for link in item.get('link') or []:
        url = clean_text(link.get('URL'))
        content_type = clean_text(link.get('content-type')).lower()
        if not url:
            continue
        pii_pdf = elsevier_pii_pdf_url(url)
        if pii_pdf:
            urls.append(pii_pdf)
            skipped.append({'url': url, 'reason': 'Elsevier XML/text endpoint replaced with ScienceDirect PDF candidate'})
            continue
        lowered = url.lower()
        if 'pdf' in content_type or lowered.endswith('.pdf') or '/pdf' in lowered or ('pdfft' in lowered):
            urls.append(url)
    if use_unpaywall:
        url = unpaywall_pdf_url(doi, unpaywall_email)
        if url:
            urls.append(url)
    if use_openalex:
        url = openalex_pdf_url(doi)
        if url:
            urls.append(url)
    if use_s2:
        url = semantic_scholar_pdf_url(doi)
        if url:
            urls.append(url)
    deduped = []
    seen = set()
    for url in urls:
        lowered = url.lower()
        domain = urlparse(url).netloc.lower()
        if 'sci-hub' in lowered or 'libgen' in lowered:
            skipped.append({'url': url, 'reason': 'pirate repository skipped'})
            continue
        skip_reason = SKIP_DOWNLOAD_DOMAINS.get(domain)
        if skip_reason:
            skipped.append({'url': url, 'reason': skip_reason})
            continue
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return (deduped, skipped)

def doi_pdf_candidates(doi: str) -> list[str]:
    doi = clean_text(doi).lower()
    if not doi:
        return []
    quoted = quote(doi, safe='/().-_:')
    suffix = doi.split('/', 1)[1] if '/' in doi else doi
    candidates: list[str] = []
    if doi.startswith('10.1007/') or doi.startswith('10.1617/'):
        candidates.append(f'https://link.springer.com/content/pdf/{quoted}.pdf')
    if doi.startswith(('10.1080/', '10.1179/', '10.1201/')):
        candidates.append(f'https://www.tandfonline.com/doi/pdf/{quoted}?download=true')
    if doi.startswith(('10.1002/', '10.1111/')):
        candidates.append(f'https://onlinelibrary.wiley.com/doi/pdfdirect/{quoted}')
        candidates.append(f'https://onlinelibrary.wiley.com/doi/pdf/{quoted}')
    if doi.startswith('10.1680/'):
        candidates.append(f'https://www.icevirtuallibrary.com/doi/pdf/{quoted}')
    if doi.startswith('10.1061/'):
        candidates.append(f'https://ascelibrary.org/doi/pdf/{quoted}')
    if doi.startswith('10.1177/'):
        candidates.append(f'https://journals.sagepub.com/doi/pdf/{quoted}')
    if doi.startswith('10.1088/'):
        candidates.append(f'https://iopscience.iop.org/article/{quoted}/pdf')
    if doi.startswith('10.1038/') and suffix:
        candidates.append(f"https://www.nature.com/articles/{quote(suffix, safe='')}.pdf")
    if doi.startswith('10.1142/'):
        candidates.append(f'https://www.worldscientific.com/doi/pdf/{quoted}')
    return candidates

def item_pdf_priority(item: dict[str, Any]) -> int:
    score = 0
    doi = clean_text(item.get('DOI')).lower()
    if doi.startswith(('10.1007/', '10.1617/', '10.4028/', '10.3151/', '10.14250/', '10.1520/', '10.1680/', '10.1155/', '10.1590/', '10.3989/', '10.21041/', '10.1088/', '10.1111/', '10.1002/')):
        score += 50
    if doi.startswith('10.1016/'):
        score -= 100
    for link in item.get('link') or []:
        url = clean_text(link.get('URL'))
        if not url:
            continue
        domain = urlparse(url).netloc.lower()
        lowered = url.lower()
        content_type = clean_text(link.get('content-type')).lower()
        if domain in SKIP_DOWNLOAD_DOMAINS:
            score -= 25
        if domain in HIGH_YIELD_DOMAINS:
            score += 40
        if 'pdf' in content_type or lowered.endswith('.pdf') or '/pdf' in lowered or ('_pdf' in lowered) or ('download' in lowered):
            score += 25
    return score

def elsevier_pii_pdf_url(url: str) -> str:
    match = re.search('/PII:([A-Za-z0-9]+)', url)
    if not match:
        return ''
    pii = match.group(1)
    return f'https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?download=true'

def safe_filename(record: Record, suffix: str='.pdf') -> str:
    base = normalize_key(record.title)[:90].replace(' ', '_')
    doi_part = normalize_key(record.doi).replace(' ', '_')[-40:]
    base = base or doi_part or record.record_id.lower()
    return f'{record.record_id}_{base}{suffix}'

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def looks_like_html(data: bytes) -> bool:
    head = data[:2048].lstrip().lower()
    return head.startswith(b'<!doctype html') or head.startswith(b'<html') or b'<title>' in head[:1024]

def validate_pdf(path: Path) -> tuple[bool, str, int]:
    try:
        with path.open('rb') as handle:
            head = handle.read(8)
        if not head.startswith(b'%PDF'):
            return (False, 'missing PDF signature', 0)
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt('')
            except Exception:
                return (False, 'encrypted PDF could not be decrypted', 0)
        page_count = len(reader.pages)
        if page_count < 1:
            return (False, 'zero pages', 0)
        return (True, 'parseable PDF', page_count)
    except Exception as exc:
        return (False, f'PDF parse failure: {exc}', 0)

def extract_pdf_text(path: Path, max_pages: int=12) -> str:
    try:
        reader = PdfReader(str(path), strict=False)
        texts = []
        for page in reader.pages[:max_pages]:
            try:
                texts.append(page.extract_text() or '')
            except Exception:
                continue
        return clean_text(' '.join(texts))
    except Exception:
        return ''

def pdf_relevance(record: Record, text: str) -> tuple[str, str]:
    haystack = normalize_key(' '.join([record.title, record.abstract, record.keywords, text]))
    has_cement = any((term in haystack for term in CEMENT_TERMS))
    title = normalize_key(record.title)
    if any((term in title for term in OUT_OF_SCOPE_TERMS)):
        return ('relevance_reject', 'out-of-scope material/application named in title')
    if has_cement and any((term in title for term in DIRECT_DEF_TERMS)):
        return ('retained', 'title explicitly identifies DEF/internal sulfate attack')
    direct_count = sum((haystack.count(term) for term in DIRECT_DEF_TERMS))
    if has_cement and direct_count >= 2:
        return ('retained', 'PDF repeatedly discusses DEF/internal sulfate attack in a cementitious system')
    ettringite_count = sum((haystack.count(term) for term in ETTRINGITE_TERMS))
    heat_count = sum((haystack.count(term) for term in EARLY_HEAT_TERMS))
    behavior_count = sum((haystack.count(term) for term in DEF_BEHAVIOR_TERMS))
    if has_cement and ettringite_count >= 3 and (heat_count >= 2) and (behavior_count >= 2):
        return ('retained', 'PDF repeatedly links cementitious ettringite, early heat, and DEF-type behavior')
    return ('relevance_reject', 'validated PDF lacks sustained direct DEF or ettringite-plus-early-heat evidence')

def curve_candidate(record: Record, text: str) -> tuple[bool, str]:
    haystack = normalize_key(' '.join([record.title, record.abstract, record.keywords, text]))
    has_curve_response = any((term in haystack for term in CURVE_TERMS))
    has_time = any((term in haystack for term in TIME_TERMS))
    has_visual_hint = any((term in haystack for term in ['figure', 'fig', 'table', 'curve', 'plotted', 'measured']))
    if has_curve_response and has_time and has_visual_hint:
        return (True, 'mentions response variable, time/age, and figure/table/curve cues')
    if has_curve_response and has_time:
        return (True, 'mentions response variable and time/age cues')
    return (False, 'no clear first-pass time-dependent expansion/strain/length-change cues')

def log_event(filename: str, data: dict[str, Any]) -> None:
    ensure_dirs()
    with (DIRS['logs'] / filename).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + '\n')

def write_tables(records: list[Record], rejects: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fieldnames = list(Record('x').row().keys())
    with (DIRS['metadata'] / 'master_metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.row())
    with (DIRS['metadata'] / 'master_metadata.json').open('w', encoding='utf-8') as handle:
        json.dump([record.row() for record in records], handle, indent=2, ensure_ascii=False)
    with (DIRS['metadata'] / 'summary_counts.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    reject_fields = ['record_id', 'title', 'doi', 'stage', 'reason', 'url', 'query_family']
    with (DIRS['metadata'] / 'rejection_log.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=reject_fields)
        writer.writeheader()
        for reject in rejects:
            writer.writerow({key: reject.get(key, '') for key in reject_fields})
    with (DIRS['metadata'] / 'rejection_log.json').open('w', encoding='utf-8') as handle:
        json.dump(rejects, handle, indent=2, ensure_ascii=False)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-valid', type=int, default=500)
    parser.add_argument('--max-records', type=int, default=10000)
    parser.add_argument('--rows-per-query', type=int, default=120)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--use-semantic-scholar', action='store_true')
    parser.add_argument('--use-openalex', action='store_true')
    parser.add_argument('--use-openalex-discovery', action='store_true')
    parser.add_argument('--use-unpaywall', action='store_true')
    parser.add_argument('--unpaywall-email', default=os.environ.get('UNPAYWALL_EMAIL', 'research-workflow@example.com'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use-existing-discovery', action='store_true')
    args = parser.parse_args()
    ensure_dirs()
    queries = build_queries(args.query_limit or None)
    queries_path = DIRS['metadata'] / 'query_families.json'
    queries_path.write_text(json.dumps([{'query_family': family, 'query': query} for family, query in queries], indent=2, ensure_ascii=False), encoding='utf-8')
    discovery_path = DIRS['metadata'] / 'crossref_raw_records.json'
    if args.use_existing_discovery and discovery_path.exists():
        print(f'Using existing discovery records from {discovery_path}', flush=True)
        items = json.loads(discovery_path.read_text(encoding='utf-8'))
    else:
        print(f'Discovering Crossref records across {len(queries)} queries...', flush=True)
        items = crossref_discover(queries, args.rows_per_query, args.max_records)
        crossref_keys = {clean_text(item.get('DOI')).lower() or normalize_key(clean_text(item.get('title'))) for item in items}
        if args.use_openalex_discovery and len(items) < args.max_records:
            remaining = args.max_records - len(items)
            print(f'Discovering OpenAlex OA records across {len(queries)} queries...', flush=True)
            items.extend(openalex_discover(queries, args.rows_per_query, remaining, crossref_keys))
    (DIRS['metadata'] / 'crossref_raw_records.json').write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding='utf-8')
    items = sorted(items, key=item_pdf_priority, reverse=True)
    previous_metadata: dict[str, dict[str, str]] = {}
    previous_path = DIRS['metadata'] / 'master_metadata.csv'
    if previous_path.exists():
        with previous_path.open(newline='', encoding='utf-8-sig') as handle:
            previous_metadata = {row['record_id']: row for row in csv.DictReader(handle)}
    records: list[Record] = []
    rejects: list[dict[str, Any]] = []
    doi_seen: dict[str, str] = {}
    title_seen: dict[str, str] = {}
    hash_seen: dict[str, str] = {}
    attempted = 0
    valid_pdf_downloads = 0
    final_retained = 0
    duplicates_removed = 0
    corrupted_or_non_pdf = 0
    relevance_rejects = 0
    for index, item in enumerate(items, start=1):
        record = record_from_crossref(item, index)
        records.append(record)
        doi_key = record.doi.lower()
        title_key = normalize_key(record.title)
        if doi_key and doi_key in doi_seen:
            record.duplicate_status = f'duplicate_metadata_doi_of={doi_seen[doi_key]}'
            duplicates_removed += 1
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'metadata_dedup', 'reason': record.duplicate_status, 'query_family': record.query_family})
            continue
        if title_key and title_key in title_seen:
            record.duplicate_status = f'duplicate_metadata_title_of={title_seen[title_key]}'
            duplicates_removed += 1
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'metadata_dedup', 'reason': record.duplicate_status, 'query_family': record.query_family})
            continue
        if doi_key:
            doi_seen[doi_key] = record.record_id
        if title_key:
            title_seen[title_key] = record.record_id
        record.duplicate_status = 'unique_metadata'
        existing_path = DIRS['valid'] / safe_filename(record)
        if existing_path.exists():
            valid, reason, page_count = validate_pdf(existing_path)
            text = extract_pdf_text(existing_path) if valid else ''
            pdf_status, pdf_reason = pdf_relevance(record, text)
            if valid and pdf_status == 'retained':
                digest = sha256_file(existing_path)
                previous = previous_metadata.get(record.record_id, {})
                record.pdf_source = previous.get('PDF_source', '')
                record.local_filename = str(existing_path.relative_to(PROJECT_ROOT))
                record.sha256 = digest
                record.page_count = str(page_count)
                record.pdf_valid = True
                record.duplicate_status = 'unique_pdf'
                record.relevance_status = 'retained'
                record.relevance_reason = pdf_reason
                record.candidate_curve, record.curve_candidate_reason = curve_candidate(record, text)
                hash_seen[digest] = record.record_id
                valid_pdf_downloads += 1
                final_retained += 1
                if final_retained >= args.target_valid:
                    break
                continue
        relevance_status, relevance_reason = metadata_relevance(record)
        record.relevance_status = relevance_status
        record.relevance_reason = relevance_reason
        if relevance_status == 'metadata_reject':
            relevance_rejects += 1
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'metadata_screening', 'reason': relevance_reason, 'query_family': record.query_family})
            continue
        urls, skipped_urls = candidate_pdf_urls(item, record.doi, use_s2=args.use_semantic_scholar, use_openalex=args.use_openalex, use_unpaywall=args.use_unpaywall, unpaywall_email=args.unpaywall_email)
        for skipped in skipped_urls:
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'pdf_discovery', 'reason': skipped['reason'], 'url': skipped['url'], 'query_family': record.query_family})
        if not urls:
            reason = 'no legal PDF URL found in Crossref/OA fallback metadata'
            if skipped_urls:
                reason = 'only skipped/unavailable PDF URLs found'
            rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'pdf_discovery', 'reason': reason, 'query_family': record.query_family})
            continue
        downloaded = False
        for url in urls:
            attempted += 1
            record.attempted_urls.append(url)
            raw_path = DIRS['raw'] / safe_filename(record)
            try:
                data, content_type = request_bytes(url)
                if len(data) < 10000:
                    raise ValueError(f'download too small ({len(data)} bytes)')
                if looks_like_html(data):
                    raise ValueError('download appears to be HTML')
                raw_path.write_bytes(data)
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                corrupted_or_non_pdf += 1
                rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'download', 'reason': repr(exc), 'url': url, 'query_family': record.query_family})
                continue
            valid, reason, page_count = validate_pdf(raw_path)
            if not valid:
                corrupted_or_non_pdf += 1
                rejected_path = DIRS['rejected'] / raw_path.name
                shutil.move(str(raw_path), str(rejected_path))
                rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'pdf_validation', 'reason': reason, 'url': url, 'query_family': record.query_family})
                continue
            digest = sha256_file(raw_path)
            if digest in hash_seen:
                duplicates_removed += 1
                record.sha256 = digest
                record.page_count = str(page_count)
                record.pdf_valid = True
                record.duplicate_status = f'duplicate_sha256_of={hash_seen[digest]}'
                rejected_path = DIRS['rejected'] / raw_path.name
                shutil.move(str(raw_path), str(rejected_path))
                rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'hash_dedup', 'reason': record.duplicate_status, 'url': url, 'query_family': record.query_family})
                downloaded = True
                break
            text = extract_pdf_text(raw_path)
            pdf_status, pdf_reason = pdf_relevance(record, text)
            if pdf_status != 'retained':
                relevance_rejects += 1
                record.relevance_status = pdf_status
                record.relevance_reason = pdf_reason
                rejected_path = DIRS['rejected'] / raw_path.name
                shutil.move(str(raw_path), str(rejected_path))
                rejects.append({'record_id': record.record_id, 'title': record.title, 'doi': record.doi, 'stage': 'pdf_relevance_screening', 'reason': pdf_reason, 'url': url, 'query_family': record.query_family})
                downloaded = True
                break
            valid_path = DIRS['valid'] / raw_path.name
            shutil.move(str(raw_path), str(valid_path))
            record.pdf_source = url
            record.local_filename = str(valid_path.relative_to(PROJECT_ROOT))
            record.sha256 = digest
            record.page_count = str(page_count)
            record.pdf_valid = True
            record.duplicate_status = 'unique_pdf'
            record.relevance_status = 'retained'
            record.relevance_reason = pdf_reason
            record.candidate_curve, record.curve_candidate_reason = curve_candidate(record, text)
            hash_seen[digest] = record.record_id
            valid_pdf_downloads += 1
            final_retained += 1
            downloaded = True
            if final_retained % 25 == 0:
                print(f'Retained {final_retained} verified PDFs...', flush=True)
                write_tables(records, rejects, build_summary(len(items), records, attempted, valid_pdf_downloads, duplicates_removed, corrupted_or_non_pdf, relevance_rejects, final_retained, args.target_valid))
            break
        if not downloaded:
            record.notes = (record.notes + '; PDF URLs attempted but none validated').strip('; ')
        if final_retained >= args.target_valid:
            break
    summary = build_summary(len(items), records, attempted, valid_pdf_downloads, duplicates_removed, corrupted_or_non_pdf, relevance_rejects, final_retained, args.target_valid)
    write_tables(records, rejects, summary)
    write_methodology(summary)
    print(json.dumps(summary, indent=2))
    return 0 if final_retained >= args.target_valid else 2

def build_summary(discovered: int, records: list[Record], attempted: int, valid_pdf_downloads: int, duplicates_removed: int, corrupted_or_non_pdf: int, relevance_rejects: int, final_retained: int, target_valid: int) -> dict[str, Any]:
    unique_metadata = sum((1 for r in records if r.duplicate_status in ('unique_metadata', 'unique_pdf')))
    candidate_count = sum((1 for r in records if r.relevance_status == 'retained' and r.candidate_curve))
    return {'records_discovered': discovered, 'unique_metadata_records_screened': unique_metadata, 'pdfs_attempted': attempted, 'valid_parseable_pdf_downloads': valid_pdf_downloads, 'duplicates_removed': duplicates_removed, 'corrupted_or_non_pdf_rejects': corrupted_or_non_pdf, 'relevance_rejects': relevance_rejects, 'final_legitimate_retained_pdfs': final_retained, 'likely_time_dependent_curve_candidates': candidate_count, 'target_reached': final_retained >= target_valid, 'target_valid_pdfs': target_valid}

def write_methodology(summary: dict[str, Any]) -> None:
    text = f'# DEF PINNs Literature Collection Workflow\n\nProject scope: delayed ettringite formation in cementitious materials, with emphasis on elevated early-age temperature or heat curing, subsequent moisture exposure, internal sulfate attack, expansion/strain/length-change kinetics, cement chemistry, mixture composition, curing regime, exposure history, and microstructure.\n\nThis folder is separate from the earlier ASR workflow. The current phase stops at literature collection, PDF validation, metadata organization, first-pass relevance screening, and preliminary curve-candidate identification. PINN modeling is intentionally not started.\n\n## Folder Map\n\n- `00_metadata`: query list, raw discovery records, master metadata CSV/JSON, summary counts, and rejection logs.\n- `01_downloads_raw`: transient download landing area used during validation.\n- `02_pdfs_validated`: unique, parseable, scientifically retained PDFs.\n- `03_rejected`: non-PDF, corrupted, duplicate, or relevance-rejected files.\n- `04_screening`: reserved for later manual/scientific screening notes.\n- `05_curve_candidates`: reserved for later curve extraction candidate sheets and figure/page manifests.\n- `scripts`: reproducible collection and validation scripts.\n- `logs`: API and download error logs.\n\n## Discovery\n\nThe collector uses Crossref journal-article discovery across multiple DEF query families and systematic term combinations. Query families include delayed ettringite formation, DEF concrete/cement, delayed ettringite expansion, ettringite reformation, secondary ettringite after heat treatment, heat-cured concrete expansion, high-temperature curing ettringite, internal sulfate attack delayed ettringite, heat treatment cementitious materials ettringite, thermal history concrete DEF, moisture exposure DEF, curing temperature DEF, sulfate-related expansion after heat curing, and combinations with expansion, strain, length change, kinetics, time, temperature, curing, cement chemistry, supplementary cementitious materials, aggregate, moisture, exposure, microstructure, concrete, mortar, and paste.\n\nPDF URLs are taken from Crossref publisher/DOI link metadata. Optional legal OA fallbacks are implemented for Unpaywall and Semantic Scholar. Pirate repositories are explicitly skipped.\n\n## Validation And Screening\n\nEach attempted file is checked for size, HTML/error-page signatures, `%PDF` signature, pypdf parseability, encryption issues, and page count. Files are deduplicated by DOI/title before download and by SHA-256 after download. Retention requires PDF text/metadata evidence of DEF or closely related internal sulfate/ettringite behavior in a cementitious-material context with heat, moisture, expansion, curing, durability, or thermal-history relevance.\n\nFirst-pass curve candidates are flagged when the PDF text/metadata contains expansion/strain/length-change/deformation cues together with time/age/kinetics and figure/table/curve cues. This is a triage flag only; later vector extraction/manual review is still required.\n\n## Latest Summary\n\n```json\n{json.dumps(summary, indent=2)}\n```\n'
    (PROJECT_ROOT / 'README.md').write_text(text, encoding='utf-8')
if __name__ == '__main__':
    raise SystemExit(main())
