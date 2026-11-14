from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / '.' / 'results'
NEURAL = REV / 'neural_models'
OUT = REV / 'reviewer_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)

def residual_summary(run: Path) -> dict[str, float | str]:
    metrics = pd.read_csv(run / 'neural_summary_metrics.csv')
    metrics = metrics[metrics['split'].eq('terminal_test') & metrics['model'].eq('pinn_shifted_first_order')].iloc[0]
    residual = pd.read_csv(run / 'neural_residual_diagnostics.csv')
    residual = residual[residual['model'].eq('pinn_shifted_first_order')].copy()
    rows: list[dict[str, float | str | int]] = []
    for (curve_id, seed, region), group in residual.groupby(['curve_id', 'seed', 'region']):
        rms_residual = float(np.sqrt(np.mean(np.square(group['residual']))))
        rms_derivative = float(np.sqrt(np.mean(np.square(group['state_derivative']))))
        rms_rhs = float(np.sqrt(np.mean(np.square(group['kinetic_rhs']))))
        normalizer = max(rms_derivative, rms_rhs, 1e-12)
        rows.append({'curve_id': curve_id, 'seed': int(seed), 'region': region, 'rms_residual': rms_residual, 'normalized_residual': rms_residual / normalizer})
    frame = pd.DataFrame(rows)
    observed = frame[frame['region'].eq('observed')]
    extrapolation = frame[frame['region'].eq('extrapolation')]
    return {'run': run.name, 'terminal_mae_percent': float(metrics['mae_mean_percent']), 'terminal_rmse_percent': float(metrics['rmse_mean_percent']), 'mean_curve_mae_percent': float(metrics['mean_curve_mae_mean_percent']), 'median_normalized_residual_observed': float(observed['normalized_residual'].median()), 'median_normalized_residual_extrapolation': float(extrapolation['normalized_residual'].median())}

def compile_compliance() -> None:
    rows = [residual_summary(run) for run in sorted(NEURAL.glob('compliance_lambda_*_5seeds'))]
    frame = pd.DataFrame(rows)
    frame['lambda_phys'] = frame['run'].str.extract('lambda_(.+)_5seeds', expand=False).str.replace('p', '.', regex=False).astype(float)
    frame.sort_values('lambda_phys').to_csv(OUT / 'review75_compliance_tradeoff.csv', index=False)

def compile_embedding() -> None:
    rows = []
    for run in sorted(NEURAL.glob('embedding_dim_*_5seeds')):
        metrics = pd.read_csv(run / 'neural_summary_metrics.csv')
        metrics = metrics[metrics['split'].eq('terminal_test')].iloc[0]
        rows.append({'embedding_dimension': int(run.name.split('_')[2]), 'terminal_mae_percent': float(metrics['mae_mean_percent']), 'terminal_rmse_percent': float(metrics['rmse_mean_percent']), 'mean_curve_mae_percent': float(metrics['mean_curve_mae_mean_percent'])})
    pd.DataFrame(rows).sort_values('embedding_dimension').to_csv(OUT / 'review75_embedding_dimension_sensitivity.csv', index=False)

def compile_shape_baseline() -> None:
    metrics = pd.read_csv(NEURAL / 'c100_shape_ic_20seeds' / 'neural_summary_metrics.csv')
    metrics = metrics[metrics['split'].eq('terminal_test')]
    metrics.to_csv(OUT / 'review75_shape_baseline_summary.csv', index=False)

def source_inference() -> None:
    errors = pd.read_csv(REV / 'compiled' / 'curve_level_seed_averaged_errors.csv')
    comparators = ['data_only', 'shifted_first_order', 'persistence', 'linear_last5', 'pinn_avrami']
    rows = []
    delete_rows = []
    for comparator in comparators:
        difference = errors[f'mae_percent_{comparator}'] - errors['mae_percent_pinn_shifted_first_order']
        source_means = errors.assign(difference=difference).groupby('source_short')['difference'].mean()
        observed = abs(float(source_means.mean()))
        permutations = []
        values = source_means.to_numpy(float)
        for mask in range(1 << len(values)):
            signs = np.array([1 if mask & 1 << j else -1 for j in range(len(values))])
            permutations.append(abs(float(np.mean(signs * values))))
        exact_p = float(np.mean(np.asarray(permutations) >= observed - 1e-15))
        nonzero = difference[~np.isclose(difference, 0.0)]
        wilcox = wilcoxon(nonzero, alternative='two-sided', zero_method='wilcox', method='approx')
        rows.append({'comparator': comparator, 'curves': len(errors), 'mean_difference_percent_point': float(difference.mean()), 'median_difference_percent_point': float(difference.median()), 'pinns_wins': int((difference > 0).sum()), 'ties': int(np.isclose(difference, 0.0).sum()), 'pinns_losses': int((difference < 0).sum()), 'source_sign_concordance': int((source_means > 0).sum()), 'source_clusters': len(source_means), 'exact_source_sign_flip_p': exact_p, 'curve_wilcoxon_w': float(wilcox.statistic), 'curve_wilcoxon_p': float(wilcox.pvalue)})
        for omitted in source_means.index:
            kept = errors[~errors['source_short'].eq(omitted)]
            kept_difference = kept[f'mae_percent_{comparator}'] - kept['mae_percent_pinn_shifted_first_order']
            delete_rows.append({'comparator': comparator, 'omitted_source': omitted, 'remaining_curves': len(kept), 'mean_difference_percent_point': float(kept_difference.mean()), 'pinns_wins': int((kept_difference > 0).sum()), 'pinns_losses': int((kept_difference < 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / 'review75_source_inference.csv', index=False)
    pd.DataFrame(delete_rows).to_csv(OUT / 'review75_delete_one_source_ranges.csv', index=False)

def chemistry_audit() -> None:
    afm = 309.0
    rows = [('gypsum', 249.6, 16), ('anhydrite', 306.2, 20), ('C-S-H/portlandite upper bound', 331.8, 18)]
    frame = pd.DataFrame(rows, columns=['route', 'solid_increment_cm3_per_mol', 'water_moles'])
    frame['increment_per_initial_afm'] = frame['solid_increment_cm3_per_mol'] / afm
    frame['internal_water_volume_cm3_per_mol'] = 18.0 * frame['water_moles']
    frame['net_increment_internal_water_cm3_per_mol'] = frame['solid_increment_cm3_per_mol'] - frame['internal_water_volume_cm3_per_mol']
    frame.to_csv(OUT / 'review75_chemistry_volume_audit.csv', index=False)

def main() -> None:
    compile_compliance()
    compile_embedding()
    compile_shape_baseline()
    source_inference()
    chemistry_audit()
    print('reviewer-75 diagnostics compiled')
if __name__ == '__main__':
    main()
