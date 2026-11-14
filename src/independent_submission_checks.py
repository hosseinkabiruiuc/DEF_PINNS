import hashlib
import itertools
import json
import re
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import expit
from scipy.stats import rankdata, wilcoxon
import sympy as sp
import pymupdf
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / '.'
REV = P / 'results'
A = REV / 'journal_check'

def norm(s):
    return re.sub('[^a-z0-9]+', ' ', unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()).strip()

def corpus():
    master = pd.read_csv(P / 'data/metadata/master_metadata.csv').fillna('')
    index = {Path(str(r.local_filename).replace('\\', '/')).name: r for r in master.itertuples() if r.local_filename}
    files = list((P / 'data/source_pdfs').glob('*.pdf')) + list((P / 'data/rejected/strict_relevance_reaudit').glob('*.pdf'))
    rows = []
    for path in sorted(files):
        m = index.get(path.name)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with pymupdf.open(path) as doc:
            text = '\n'.join((page.get_text() for page in doc))
            first = '\n'.join((doc[i].get_text() for i in range(min(2, len(doc)))))
            nt = norm(first)
            title = norm(m.title) if m else ''
            words = set(title.split())
            overlap = len(words & set(nt.split())) / max(len(words), 1)
            rows.append({'file': path.name, 'retained': path.parent.name == 'data/source_pdfs', 'sha256': digest, 'metadata_hash_match': bool(m is not None and str(m.SHA256) == digest), 'pages': len(doc), 'text_characters': len(text), 'title_token_coverage': overlap, 'doi_on_first_pages': bool(m is not None and str(m.DOI).lower() in first.lower()), 'direct_def_mentions': len(re.findall('delayed\\s+ettringite|internal\\s+sul(?:f|ph)ate', text, re.I)), 'supplement_title_flag': bool(re.search('^\\s*(?:supplementary|supporting)\\s+(?:information|material)', first, re.I)), 'title': str(m.title) if m else ''})
        if len(rows) % 50 == 0:
            print('Corpus audit', len(rows), '/', len(files), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(A / 'full_corpus_identity_check.csv', index=False)
    flags = frame[frame.retained & ((frame.title_token_coverage < 0.55) | frame.direct_def_mentions.eq(0) | frame.supplement_title_flag)]
    flags.to_csv(A / 'retained_identity_flags.csv', index=False)
    result = {'all_pdfs': len(frame), 'retained': int(frame.retained.sum()), 'quarantined': int((~frame.retained).sum()), 'all_hashes_unique': not frame.sha256.duplicated().any(), 'all_metadata_hashes_match': bool(frame.metadata_hash_match.all()), 'retained_flags_for_review': len(flags)}
    (A / 'full_corpus_identity_summary.json').write_text(json.dumps(result, indent=2))
    print(result)

def equations():
    t, t0, s, k, a, n = sp.symbols('t t0 s k a n', positive=True)
    clock = (sp.log(1 + sp.exp(s * (t - t0))) - sp.log(1 + sp.exp(-s * t0))) / s
    gate = 1 / (1 + sp.exp(-s * (t - t0)))
    fo = a * (1 - sp.exp(-k * clock))
    av = a * (1 - sp.exp(-(k * clock) ** n))
    u = sp.symbols('u', positive=True)
    generic_av = a * (1 - sp.exp(-(k * u) ** n))
    identities = {'clock_initial': sp.simplify(clock.subs(t, 0)) == 0, 'clock_derivative': sp.simplify(sp.diff(clock, t) - gate) == 0, 'first_order_ode': sp.simplify(sp.diff(fo, t) - gate * k * (a - fo)) == 0, 'avrami_chain_rule_positive_clock': sp.simplify(sp.diff(generic_av, u) - n * k * (k * u) ** (n - 1) * (a - generic_av)) == 0, 'avrami_n1': sp.simplify(av.subs(n, 1) - fo) == 0}
    cases = []
    for onset, rate, exponent in itertools.product([0.0, 0.3, 1.5], [0.02, 0.5, 3.0], [1.0, 2.0, 4.0]):

        def om(x):
            return (np.logaddexp(0.0, 200 * (x - onset)) - np.logaddexp(0.0, -200 * onset)) / 200
        times = np.linspace(0, 3, 121)
        exact = 1 - np.exp(-(rate * om(times)) ** exponent)

        def rhs(x, y):
            u = max(rate * om(x), 0.0)
            return expit(200 * (x - onset)) * exponent * rate * u ** (exponent - 1) * (1 - y)
        sol = solve_ivp(rhs, (0, 3), [0.0], t_eval=times, rtol=1e-10, atol=1e-12, max_step=0.01)
        cases.append(float(np.max(abs(exact - sol.y[0]))))
    afm = np.array([4, 1, 1, 12])
    et = np.array([6, 1, 3, 32])
    identities['gypsum_balance'] = bool(np.array_equal(afm + 2 * np.array([1, 0, 1, 2]) + [0, 0, 0, 16], et))
    identities['anhydrite_balance'] = bool(np.array_equal(afm + 2 * np.array([1, 0, 1, 0]) + [0, 0, 0, 20], et))
    result = {'identities': identities, 'independent_ode_cases': len(cases), 'maximum_state_error': max(cases)}
    (A / 'independent_equation_checks.json').write_text(json.dumps(result, indent=2))
    print(result)
    assert all(identities.values()) and max(cases) < 1e-08

def metrics():
    obs = pd.read_csv(P / 'data/modeling/def_kinetics_observations_v2.csv')
    pred = pd.read_csv(REV / 'neural_models/primary_selected_20seeds/neural_predictions.csv')
    assert not pred.duplicated(['model', 'seed', 'observation_id']).any()
    assert len(pred) == 1725 * 20 * 3
    assert np.isfinite(pred.predicted_expansion_percent).all()
    train = obs[obs.split.eq('train')]
    test = obs[obs.split.eq('terminal_test')]
    assert (test.groupby('curve_id').time_days.min() > train.groupby('curve_id').time_days.max()).all()
    selected = pred[pred.split.eq('terminal_test')].copy()
    selected['ae'] = abs(selected.predicted_expansion_percent - selected.observed_expansion_percent)
    selected['se'] = (selected.predicted_expansion_percent - selected.observed_expansion_percent) ** 2
    curve = selected.groupby(['model', 'seed', 'curve_id']).agg(mae=('ae', 'mean'), mse=('se', 'mean')).reset_index()
    curve['rmse'] = np.sqrt(curve.mse)
    curve = curve.groupby(['model', 'curve_id'])[['mae', 'rmse']].mean()
    table = pd.read_csv(REV / 'compiled/curve_level_statistical_tests.csv')
    checks = []
    wide = pd.read_csv(REV / 'compiled/curve_level_seed_averaged_errors.csv').set_index('curve_id')
    for row in table.itertuples():
        metric = row.metric.replace('_percent', '')

        def values(model):
            if model in pred.model.unique():
                return curve.loc[model, metric].reindex(wide.index).to_numpy()
            return wide[metric + '_percent_' + model].to_numpy()
        c, b = (values(row.candidate), values(row.comparator))
        d = b - c
        stat = wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='auto')
        ranks = rankdata(abs(d[d != 0]))
        signs = d[d != 0] > 0
        rb = (ranks[signs].sum() - ranks[~signs].sum()) / ranks.sum()
        checks.append({'comparison': row.comparison, 'metric': metric, 'reduction': 100 * d.mean() / b.mean(), 'p': float(stat.pvalue), 'W': float(stat.statistic), 'r': float(rb), 'matches': bool(np.isclose(100 * d.mean() / b.mean(), row.relative_reduction_percent, atol=1e-09) and np.isclose(stat.pvalue, row.raw_two_sided_p, atol=1e-12) and np.isclose(rb, row.rank_biserial_correlation, atol=1e-12))})
    pvalues = np.array([x['p'] for x in checks])
    order = np.argsort(pvalues)
    corrected = np.minimum(1.0, np.maximum.accumulate((len(order) - np.arange(len(order))) * pvalues[order]))
    holm = np.empty_like(corrected)
    holm[order] = corrected
    assert np.allclose(holm, table.holm_adjusted_p, rtol=1e-09, atol=1e-12)
    assert all((x['matches'] for x in checks))
    (A / 'independent_metric_checks.json').write_text(json.dumps({'comparisons': checks, 'holm_matches': True, 'prediction_rows': len(pred), 'no_temporal_overlap': True}, indent=2))
    print('Independent prediction-level inference passed')
if __name__ == '__main__':
    import sys
    {'corpus': corpus, 'equations': equations, 'metrics': metrics}[sys.argv[1]]()
