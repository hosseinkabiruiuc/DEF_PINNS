from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / 'data/extracted'
OUT = ROOT / 'data/modeling'
INPUTS = ['pavoine2007_vector_curves.csv', 'tiecher2021_vector_curves.csv', 'aubert2013_vector_curves.csv', 'martin2023_vector_curves.csv', 'kawabata2023_vector_curves.csv']
MOISTURE_INDEX = {'20 C, 100% RH after cyclic wetting/drying': 0.75, 'continuous_water_immersion': 1.0, 'frequently_renewed_deionised_water': 1.0, 'continuous_immersion': 1.0, 'wetting_drying': 0.3, 'partial_tap_water_immersion': 0.8}

def numeric_column(frame: pd.DataFrame, name: str, default: float=np.nan) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors='coerce')

def uniformly_cap(group: pd.DataFrame, maximum: int=60) -> pd.DataFrame:
    group = group.sort_values('time_days').drop_duplicates(subset=['time_days', 'expansion_percent'])
    if len(group) <= maximum:
        return group
    indices = np.unique(np.rint(np.linspace(0, len(group) - 1, maximum)).astype(int))
    return group.iloc[indices]

def curve_quality_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for curve_id, group in data.groupby('curve_id', sort=True):
        ordered = group.sort_values('time_days')
        expansion = ordered['expansion_percent'].to_numpy(dtype=float)
        downward_steps = np.maximum(-np.diff(expansion), 0.0)
        maximum_downward_step = float(downward_steps.max()) if len(downward_steps) else 0.0
        total_downward_variation = float(downward_steps.sum())
        is_pavoine = str(ordered['record_id'].iloc[0]) == 'DEF-000170'
        ambiguous = is_pavoine and maximum_downward_step > 0.02
        rows.append({'curve_id': curve_id, 'record_id': ordered['record_id'].iloc[0], 'raw_observations': len(ordered), 'maximum_downward_step_percent': maximum_downward_step, 'total_downward_variation_percent': total_downward_variation, 'accepted_for_modeling': not ambiguous, 'quality_reason': 'excluded: overlapping grayscale marker styles produce a nonphysical branch' if ambiguous else 'accepted: no source-specific trajectory ambiguity detected'})
    return pd.DataFrame(rows)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for filename in INPUTS:
        frame = pd.read_csv(SOURCE_DIR / filename)
        frame['source_data_file'] = filename
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    raw_points = len(data)
    data['time_days_raw'] = numeric_column(data, 'time_days')
    data['time_days'] = data['time_days_raw'].clip(lower=0.0)
    data['expansion_percent_raw'] = numeric_column(data, 'expansion_percent')
    data['expansion_percent'] = data['expansion_percent_raw'].clip(lower=0.0)
    data['heat_temperature_C'] = numeric_column(data, 'temperature_C')
    data['heat_duration_days'] = numeric_column(data, 'treatment_duration_days')
    data['storage_temperature_C_std'] = numeric_column(data, 'storage_temperature_C')
    data['storage_temperature_C_std'] = data['storage_temperature_C_std'].fillna(numeric_column(data, 'post_cure_exposure_C'))
    data['storage_temperature_C_std'] = data['storage_temperature_C_std'].fillna(20.0)
    data['water_cement_ratio_std'] = numeric_column(data, 'water_cement_ratio')
    data['SO3_percent_std'] = numeric_column(data, 'SO3_percent')
    data['Al2O3_percent_std'] = numeric_column(data, 'Al2O3_percent')
    data['SO3_Al2O3_ratio_std'] = numeric_column(data, 'SO3_Al2O3_ratio')
    ratio = data['SO3_percent_std'] / data['Al2O3_percent_std']
    data['SO3_Al2O3_ratio_std'] = data['SO3_Al2O3_ratio_std'].fillna(ratio)
    data['alkali_percent_std'] = numeric_column(data, 'alkali_percent')
    data['alkali_percent_std'] = data['alkali_percent_std'].fillna(numeric_column(data, 'Na2Oeq_percent'))
    data['alkali_percent_std'] = data['alkali_percent_std'].fillna(numeric_column(data, 'Na2O_percent'))
    data['scm_percent_std'] = numeric_column(data, 'scm_percent', 0.0).fillna(0.0)
    data['moisture_index'] = data['moisture_condition'].map(MOISTURE_INDEX).fillna(0.75)
    data['restraint_index'] = 0.0
    kawabata = data['record_id'].eq('DEF-001010')
    data.loc[kawabata, 'restraint_index'] = 0.35
    data.loc[kawabata & numeric_column(data, 'reinforcement_stirrups', 0.0).eq(1.0), 'restraint_index'] = 1.0
    data = data[data['quality_status'].eq('accepted') & data['time_days'].notna() & data['expansion_percent'].notna() & data['axis_calibration_r2_time'].astype(float).ge(0.997) & data['axis_calibration_r2_expansion'].astype(float).ge(0.997)].copy()
    quality_audit = curve_quality_audit(data)
    accepted_quality_curves = quality_audit.loc[quality_audit['accepted_for_modeling'], 'curve_id']
    data = data[data['curve_id'].isin(accepted_quality_curves)].copy()
    data = pd.concat([uniformly_cap(group) for _, group in data.groupby('curve_id', sort=True)], ignore_index=True)
    curve_counts = data.groupby('curve_id').size()
    accepted_curves = curve_counts[curve_counts >= 6].index
    data = data[data['curve_id'].isin(accepted_curves)].copy()
    data['source_code'] = pd.Categorical(data['record_id']).codes
    data['curve_code'] = pd.Categorical(data['curve_id']).codes
    data['split'] = 'train'
    data['terminal_rank'] = 0
    for curve_id, indices in data.groupby('curve_id').groups.items():
        ordered = data.loc[indices].sort_values('time_days')
        holdout_count = max(2, int(math.ceil(0.2 * len(ordered))))
        holdout_indices = ordered.index[-holdout_count:]
        data.loc[holdout_indices, 'split'] = 'terminal_test'
        data.loc[ordered.index, 'terminal_rank'] = np.arange(1, len(ordered) + 1)
    data = data.sort_values(['record_id', 'curve_id', 'time_days']).reset_index(drop=True)
    data['observation_id'] = [f'DEF-OBS-{index:06d}' for index in range(1, len(data) + 1)]
    curve_rows = []
    for curve_id, group in data.groupby('curve_id', sort=True):
        ordered = group.sort_values('time_days')
        terminal = float(ordered.iloc[-1]['expansion_percent'])
        response_class = 'negligible' if terminal < 0.04 else 'moderate' if terminal < 0.6 else 'large'
        curve_rows.append({'curve_id': curve_id, 'curve_code': int(group['curve_code'].iloc[0]), 'record_id': group['record_id'].iloc[0], 'source_short': group['source_short'].iloc[0], 'figure_id': group['figure_id'].iloc[0], 'observations': len(group), 'train_observations': int(group['split'].eq('train').sum()), 'terminal_test_observations': int(group['split'].eq('terminal_test').sum()), 'maximum_time_days': float(ordered.iloc[-1]['time_days']), 'terminal_expansion_percent': terminal, 'response_class': response_class, 'heat_temperature_C': float(group['heat_temperature_C'].iloc[0]), 'heat_duration_days': float(group['heat_duration_days'].iloc[0]), 'moisture_index': float(group['moisture_index'].iloc[0]), 'restraint_index': float(group['restraint_index'].iloc[0])})
    curves = pd.DataFrame(curve_rows)
    data.to_csv(OUT / 'def_kinetics_observations_v2.csv', index=False)
    curves.to_csv(OUT / 'def_kinetics_curves_v2.csv', index=False)
    quality_audit.to_csv(OUT / 'def_kinetics_curve_quality_v2.csv', index=False)
    summary = {'dataset_version': 'v2', 'source_studies': int(data['record_id'].nunique()), 'curves': int(data['curve_id'].nunique()), 'raw_extracted_observations': raw_points, 'accepted_observations_after_uniform_cap': len(data), 'train_observations': int(data['split'].eq('train').sum()), 'terminal_test_observations': int(data['split'].eq('terminal_test').sum()), 'time_range_days': [float(data['time_days'].min()), float(data['time_days'].max())], 'expansion_range_percent_raw': [float(data['expansion_percent_raw'].min()), float(data['expansion_percent_raw'].max())], 'response_classes': curves['response_class'].value_counts().to_dict(), 'study_curve_counts': curves.groupby('record_id').size().to_dict(), 'quality_rules': {'axis_calibration_r2_minimum': 0.997, 'minimum_points_per_curve': 6, 'maximum_points_per_curve': 60, 'terminal_holdout_fraction': 0.2, 'negative_measurement_noise_clipped_for_physical_model': True, 'brunetaud_maximum_downward_step_percent': 0.02, 'brunetaud_ambiguous_style_groups_excluded': int((~quality_audit['accepted_for_modeling']).sum())}}
    (OUT / 'def_kinetics_dataset_summary_v2.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
