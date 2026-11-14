from __future__ import annotations
import argparse
import json
import math
import random
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/modeling' / 'def_kinetics_observations_v2.csv'
OUT = ROOT / 'results' / 'neural_models'
FEATURES = ['heat_temperature_C', 'heat_duration_days', 'storage_temperature_C_std', 'water_cement_ratio_std', 'SO3_Al2O3_ratio_std', 'alkali_percent_std', 'scm_percent_std', 'moisture_index', 'restraint_index']
MODEL_NAMES = ['data_only', 'shape_regularized', 'pinn_shifted_first_order', 'pinn_avrami']

def integrated_activation(tau: torch.Tensor, onset: torch.Tensor, slope: float) -> torch.Tensor:
    clock = (torch.nn.functional.softplus(slope * (tau - onset)) - torch.nn.functional.softplus(-slope * onset)) / slope
    return clock.clamp_min(1e-12)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current, width), nn.Tanh()])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)

class ParametricStateNet(nn.Module):

    def __init__(self, curves: int, descriptor_dim: int, embedding_dim: int, onset_upper: np.ndarray, width: int, depth: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(curves, embedding_dim) if embedding_dim > 0 else None
        self.register_buffer('onset_upper', torch.tensor(onset_upper, dtype=torch.float32).reshape(-1, 1))
        context_dim = descriptor_dim + embedding_dim
        self.parameter_net = mlp(context_dim, 4, width=48, depth=2)
        self.state_net = mlp(1 + context_dim + 4, 1, width=width, depth=depth)
        final_layer = self.state_net[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.constant_(final_layer.bias, -4.0)

    def context(self, curve_index: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            return descriptors
        return torch.cat([self.embedding(curve_index), descriptors], dim=1)

    def forward(self, tau: torch.Tensor, curve_index: torch.Tensor, descriptors: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.context(curve_index, descriptors)
        raw = self.parameter_net(context)
        amplitude = 1.5 * torch.sigmoid(raw[:, 0:1])
        rate = 0.02 + 30.0 * torch.sigmoid(raw[:, 1:2])
        onset = self.onset_upper[curve_index] * torch.sigmoid(raw[:, 2:3])
        exponent = 1.0 + 3.0 * torch.sigmoid(raw[:, 3:4])
        parameters = torch.cat([amplitude, rate, onset, exponent], dim=1)
        state = torch.nn.functional.softplus(self.state_net(torch.cat([tau, context, parameters], dim=1)))
        return (state, {'amplitude': amplitude, 'rate': rate, 'onset': onset, 'exponent': exponent})

def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {'mae_percent': float(np.mean(np.abs(residual))), 'rmse_percent': float(np.sqrt(np.mean(residual ** 2))), 'r2': 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else math.nan, 'holdout_variance': float(np.var(observed, ddof=0))}

def prepare_data(args: argparse.Namespace) -> dict[str, object]:
    data = pd.read_csv(DATA).sort_values(['curve_code', 'time_days']).reset_index(drop=True)
    if args.exclude_source_record_id:
        data = data[data['record_id'] != args.exclude_source_record_id].copy().reset_index(drop=True)
    curve_labels = sorted(data['curve_id'].unique())
    curve_lookup = {curve_id: index for index, curve_id in enumerate(curve_labels)}
    data['curve_code_v3'] = data['curve_id'].map(curve_lookup).astype(int)
    curve_frame = data.groupby('curve_code_v3', sort=True).first().reset_index()
    features = [feature for feature in FEATURES if not (args.drop_sulfate_ratio and feature == 'SO3_Al2O3_ratio_std')]
    descriptor_raw = curve_frame[features].astype(float)
    missing = descriptor_raw.isna().astype(float)
    source_train_curves = ~curve_frame['record_id'].eq(args.holdout_source_record_id) if args.holdout_source_record_id else np.ones(len(curve_frame), dtype=bool)
    medians = descriptor_raw.loc[source_train_curves].median(axis=0)
    imputed = descriptor_raw.fillna(medians)
    means = imputed.loc[source_train_curves].mean(axis=0)
    stds = imputed.loc[source_train_curves].std(axis=0).replace(0.0, 1.0)
    standardized = (imputed - means) / stds
    curve_descriptors = np.concatenate([standardized.to_numpy(np.float32), missing.to_numpy(np.float32)], axis=1)
    curve_indices = data['curve_code_v3'].to_numpy(np.int64)
    original_train = data['split'].eq('train').to_numpy()
    base_train = original_train.copy()
    if args.holdout_source_record_id:
        base_train &= ~data['record_id'].eq(args.holdout_source_record_id).to_numpy()
    candidate_train = base_train.copy()
    evaluation_split = data['split'].astype(str).to_numpy()
    if not 0.0 < args.train_observation_fraction <= 1.0:
        raise ValueError('train_observation_fraction must be in (0, 1]')
    if args.train_observation_fraction < 1.0:
        candidate_train[:] = False
        for _, group in data[data['split'].eq('train')].groupby('curve_code_v3', sort=True):
            group = group.sort_values('time_days')
            t_min = float(group['time_days'].min())
            t_max = float(group['time_days'].max())
            cutoff = t_min + args.train_observation_fraction * (t_max - t_min)
            kept = group[group['time_days'] <= cutoff]
            if len(kept) < min(5, len(group)):
                kept = group.iloc[:min(5, len(group))]
            candidate_train[kept.index] = True
            dropped = group.index.difference(kept.index)
            evaluation_split[dropped] = 'preterminal_unobserved'
    train_mask = candidate_train.copy()
    if not 0.0 <= args.internal_validation_fraction < 0.5:
        raise ValueError('internal_validation_fraction must be in [0, 0.5)')
    if args.internal_validation_fraction > 0.0:
        for _, group in data[candidate_train].groupby('curve_code_v3', sort=True):
            group = group.sort_values('time_days')
            validation_count = max(1, int(math.ceil(args.internal_validation_fraction * len(group))))
            validation_count = min(validation_count, max(1, len(group) - 3))
            validation_indices = group.index[-validation_count:]
            train_mask[validation_indices] = False
            evaluation_split[group.index] = 'fit_train'
            evaluation_split[validation_indices] = 'internal_validation'
    curve_count = len(curve_frame)
    full_time_scale = data.groupby('curve_code_v3')['time_days'].max().sort_index().to_numpy(float)
    training_time_scale = np.zeros(curve_count, dtype=float)
    for curve_code in range(curve_count):
        curve_mask = data['curve_code_v3'] == curve_code
        values = data.loc[train_mask & curve_mask, 'time_days']
        if values.empty and args.holdout_source_record_id:
            values = data.loc[original_train & curve_mask, 'time_days']
        if values.empty:
            raise ValueError(f'curve {curve_code} has no training observations')
        training_time_scale[curve_code] = float(values.max())
    time_scale = full_time_scale if args.time_scaling == 'full_series' else training_time_scale
    time_scale = np.maximum(time_scale, 1.0)
    tau = data['time_days'].to_numpy(np.float32) / time_scale[curve_indices].astype(np.float32)
    if args.amplitude_scaling == 'full_series':
        y_scale = float(max(data['expansion_percent'].max(), 0.1))
    else:
        y_scale = float(max(data.loc[train_mask, 'expansion_percent'].max(), 0.1))
    targets = data['expansion_percent'].to_numpy(np.float32) / y_scale
    curve_scales = np.full(curve_count, y_scale, dtype=np.float32)
    if args.amplitude_scaling == 'per_curve_training':
        for curve_code in range(curve_count):
            values = data.loc[train_mask & (data['curve_code_v3'] == curve_code), 'expansion_percent']
            if values.empty:
                raise ValueError('Per-curve response scaling is unavailable for an unseen source')
            curve_scales[curve_code] = max(float(values.max()), 0.015)
        targets = data['expansion_percent'].to_numpy(np.float32) / curve_scales[curve_indices]
    onset_upper = np.full(curve_count, args.onset_max_tau, dtype=np.float32)
    for curve_code in range(curve_count):
        rows = data.loc[train_mask & (data['curve_code_v3'] == curve_code)].sort_values('time_days')
        activated = rows[rows['expansion_percent'] >= args.activation_detection_percent]
        if not activated.empty:
            first_detectable_tau = float(activated.iloc[0]['time_days']) / float(time_scale[curve_code])
            onset_upper[curve_code] = np.float32(np.clip(first_detectable_tau + args.activation_margin_tau, 0.15, args.onset_max_tau))
    train_curve = curve_indices[train_mask]
    counts = np.bincount(train_curve, minlength=curve_count).astype(np.float32)
    weights = 1.0 / np.maximum(counts[train_curve], 1.0)
    weights /= weights.mean()
    return {'data': data, 'curve_frame': curve_frame, 'curve_labels': curve_labels, 'curve_descriptors': curve_descriptors, 'curve_indices': curve_indices, 'time_scale': time_scale, 'training_time_scale': training_time_scale, 'full_time_scale': full_time_scale, 'tau': tau, 'targets': targets, 'onset_upper': onset_upper, 'train_mask': train_mask, 'train_weights': weights, 'evaluation_split': evaluation_split, 'y_scale': y_scale, 'curve_scales': curve_scales, 'feature_medians': medians.to_dict(), 'feature_means': means.to_dict(), 'feature_stds': stds.to_dict(), 'features': features}

def train_one(model_name: str, seed: int, prepared: dict[str, object], args: argparse.Namespace) -> tuple[ParametricStateNet, list[dict[str, float]]]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    descriptors_np = prepared['curve_descriptors']
    train_mask = prepared['train_mask']
    curve_indices = prepared['curve_indices']
    tau = prepared['tau']
    targets = prepared['targets']
    assert all((isinstance(x, np.ndarray) for x in [descriptors_np, train_mask, curve_indices, tau, targets]))
    curve_count = descriptors_np.shape[0]
    embedding_dim = 0 if args.no_embedding else args.embedding_dim
    model = ParametricStateNet(curve_count, descriptors_np.shape[1], embedding_dim, prepared['onset_upper'], args.width, args.depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate / 10.0)
    train_curve = torch.tensor(curve_indices[train_mask], dtype=torch.long)
    train_tau = torch.tensor(tau[train_mask, None], dtype=torch.float32)
    train_y = torch.tensor(targets[train_mask, None], dtype=torch.float32)
    train_desc = torch.tensor(descriptors_np[curve_indices[train_mask]], dtype=torch.float32)
    train_weights = torch.tensor(prepared['train_weights'], dtype=torch.float32).reshape(-1, 1)
    coll_curve = torch.arange(curve_count, dtype=torch.long).repeat_interleave(args.collocation_count)
    coll_desc = torch.tensor(descriptors_np[coll_curve.numpy()], dtype=torch.float32)
    init_curve = torch.arange(curve_count, dtype=torch.long)
    init_tau = torch.zeros((curve_count, 1), dtype=torch.float32)
    init_desc = torch.tensor(descriptors_np, dtype=torch.float32)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = model(train_tau, train_curve, train_desc)
        data_loss = torch.sum(train_weights * (predicted - train_y) ** 2) / torch.sum(train_weights)
        physics_loss = torch.tensor(0.0)
        monotonic_loss = torch.tensor(0.0)
        initial_loss = torch.tensor(0.0)
        if model_name != 'data_only':
            cells = torch.arange(args.collocation_count, dtype=torch.float32)[None, :]
            jitter = torch.rand((curve_count, args.collocation_count), dtype=torch.float32)
            coll_tau = ((cells + jitter) / args.collocation_count * args.collocation_max_tau).reshape(-1, 1)
            if args.collocation_sampling == 'gate_aware':
                with torch.no_grad():
                    _, head = model(init_tau, init_curve, init_desc)
                    near_count = args.collocation_count // 2
                    near = head['onset'] + (2 * torch.rand((curve_count, near_count)) - 1) * 4 / args.gate_slope
                    sampled = coll_tau.reshape(curve_count, args.collocation_count)
                    sampled[:, -near_count:] = near.clamp(0, args.collocation_max_tau)
                    coll_tau = sampled.reshape(-1, 1)
            coll_tau.requires_grad_(True)
            coll_state, parameters = model(coll_tau, coll_curve, coll_desc)
            derivative = torch.autograd.grad(coll_state.sum(), coll_tau, create_graph=True)[0]
            gate = torch.sigmoid(args.gate_slope * (coll_tau - parameters['onset']))
            if model_name == 'pinn_shifted_first_order':
                rhs = gate * parameters['rate'] * (parameters['amplitude'] - coll_state)
            elif model_name == 'pinn_avrami':
                shifted = integrated_activation(coll_tau, parameters['onset'], args.gate_slope)
                rhs = gate * parameters['exponent'] * parameters['rate'] * torch.pow(parameters['rate'] * shifted, parameters['exponent'] - 1.0) * (parameters['amplitude'] - coll_state)
            else:
                rhs = derivative.detach()
            physics_loss = torch.mean((derivative - rhs) ** 2)
            monotonic_loss = torch.mean(torch.relu(-derivative) ** 2)
            initial_state, _ = model(init_tau, init_curve, init_desc)
            initial_loss = torch.mean(initial_state ** 2)
        total_loss = data_loss + args.physics_weight * physics_loss + args.monotonicity_weight * monotonic_loss + args.initial_weight * initial_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % args.history_interval == 0 or epoch == args.epochs:
            history.append({'epoch': epoch, 'total_loss': float(total_loss.detach()), 'data_loss': float(data_loss.detach()), 'physics_loss_raw': float(physics_loss.detach()), 'physics_loss_weighted': float((args.physics_weight * physics_loss).detach()), 'monotonic_loss_raw': float(monotonic_loss.detach()), 'monotonic_loss_weighted': float((args.monotonicity_weight * monotonic_loss).detach()), 'initial_loss_raw': float(initial_loss.detach()), 'initial_loss_weighted': float((args.initial_weight * initial_loss).detach()), 'learning_rate': float(scheduler.get_last_lr()[0])})
    return (model, history)

def residual_diagnostics(model: ParametricStateNet, model_name: str, seed: int, prepared: dict[str, object], args: argparse.Namespace) -> list[dict[str, object]]:
    if model_name not in {'pinn_shifted_first_order', 'pinn_avrami', 'shape_regularized'}:
        return []
    descriptors_np = prepared['curve_descriptors']
    curve_labels = prepared['curve_labels']
    curve_frame = prepared['curve_frame']
    assert isinstance(descriptors_np, np.ndarray)
    rows: list[dict[str, object]] = []
    grid = torch.linspace(0.0, args.collocation_max_tau, 121, dtype=torch.float32)
    for curve_index, curve_id in enumerate(curve_labels):
        tau = grid[:, None].clone().requires_grad_(True)
        curve = torch.full((len(grid),), curve_index, dtype=torch.long)
        desc = torch.tensor(np.repeat(descriptors_np[curve_index][None, :], len(grid), axis=0), dtype=torch.float32)
        state, parameters = model(tau, curve, desc)
        derivative = torch.autograd.grad(state.sum(), tau, create_graph=False)[0]
        gate = torch.sigmoid(args.gate_slope * (tau - parameters['onset']))
        if model_name in {'pinn_shifted_first_order', 'shape_regularized'}:
            rhs = gate * parameters['rate'] * (parameters['amplitude'] - state)
        else:
            shifted = integrated_activation(tau, parameters['onset'], args.gate_slope)
            rhs = gate * parameters['exponent'] * parameters['rate'] * torch.pow(parameters['rate'] * shifted, parameters['exponent'] - 1.0) * (parameters['amplitude'] - state)
        residual = derivative - rhs
        source = str(curve_frame.iloc[curve_index]['source_short'])
        for tau_value, derivative_value, rhs_value, residual_value in zip(grid.numpy(), derivative.detach().numpy().ravel(), rhs.detach().numpy().ravel(), residual.detach().numpy().ravel()):
            rows.append({'curve_id': curve_id, 'source_short': source, 'model': model_name, 'seed': seed, 'tau': float(tau_value), 'region': 'observed' if tau_value <= 1.0 else 'extrapolation', 'state_derivative': float(derivative_value), 'kinetic_rhs': float(rhs_value), 'residual': float(residual_value)})
    return rows

def evaluate(model: ParametricStateNet, model_name: str, seed: int, prepared: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data = prepared['data']
    descriptors_np = prepared['curve_descriptors']
    curve_indices = prepared['curve_indices']
    tau = prepared['tau']
    assert isinstance(data, pd.DataFrame)
    assert all((isinstance(x, np.ndarray) for x in [descriptors_np, curve_indices, tau]))
    y_scale = float(prepared['y_scale'])
    with torch.no_grad():
        state, _ = model(torch.tensor(tau[:, None], dtype=torch.float32), torch.tensor(curve_indices, dtype=torch.long), torch.tensor(descriptors_np[curve_indices], dtype=torch.float32))
        predictions = state[:, 0].numpy() * prepared['curve_scales'][curve_indices]
    prediction_rows: list[dict[str, object]] = []
    for index, source_row in data.iterrows():
        prediction_rows.append({'observation_id': source_row['observation_id'], 'curve_id': source_row['curve_id'], 'record_id': source_row['record_id'], 'source_short': source_row['source_short'], 'model': model_name, 'seed': seed, 'split': prepared['evaluation_split'][index], 'time_days': source_row['time_days'], 'tau': float(tau[index]), 'observed_expansion_percent': source_row['expansion_percent'], 'predicted_expansion_percent': float(predictions[index])})
    prediction_frame = pd.DataFrame(prediction_rows)
    metric_rows: list[dict[str, object]] = []
    for split_name, split_group in prediction_frame.groupby('split'):
        global_metrics = regression_metrics(split_group['observed_expansion_percent'].to_numpy(), split_group['predicted_expansion_percent'].to_numpy())
        curve_metrics = [regression_metrics(group['observed_expansion_percent'].to_numpy(), group['predicted_expansion_percent'].to_numpy()) for _, group in split_group.groupby('curve_id')]
        metric_rows.append({'model': model_name, 'seed': seed, 'split': split_name, 'observations': len(split_group), **global_metrics, 'mean_curve_mae_percent': float(np.mean([row['mae_percent'] for row in curve_metrics])), 'median_curve_mae_percent': float(np.median([row['mae_percent'] for row in curve_metrics])), 'mean_curve_rmse_percent': float(np.mean([row['rmse_percent'] for row in curve_metrics]))})
    curve_frame = prepared['curve_frame']
    time_scale = prepared['time_scale']
    training_time_scale = prepared['training_time_scale']
    full_time_scale = prepared['full_time_scale']
    onset_upper = prepared['onset_upper']
    assert isinstance(curve_frame, pd.DataFrame)
    assert all((isinstance(x, np.ndarray) for x in [time_scale, training_time_scale, full_time_scale, onset_upper]))
    curve_codes = torch.arange(len(curve_frame), dtype=torch.long)
    with torch.no_grad():
        _, parameters = model(torch.full((len(curve_frame), 1), 0.5, dtype=torch.float32), curve_codes, torch.tensor(descriptors_np, dtype=torch.float32))
    parameter_rows: list[dict[str, object]] = []
    for index, curve_row in curve_frame.iterrows():
        parameter_rows.append({'curve_id': curve_row['curve_id'], 'record_id': curve_row['record_id'], 'source_short': curve_row['source_short'], 'model': model_name, 'seed': seed, 'time_scale_days': float(time_scale[index]), 'training_horizon_days': float(training_time_scale[index]), 'full_horizon_days': float(full_time_scale[index]), 'onset_upper_normalized': float(onset_upper[index]), 'amplitude_percent': float(parameters['amplitude'][index]) * float(prepared['curve_scales'][index]), 'amplitude_scale_percent': float(prepared['curve_scales'][index]), 'rate_normalized': float(parameters['rate'][index]), 'rate_per_day': float(parameters['rate'][index]) / float(time_scale[index]), 'onset_normalized': float(parameters['onset'][index]), 'onset_days': float(parameters['onset'][index]) * float(time_scale[index]), 'avrami_exponent': float(parameters['exponent'][index])})
    return (prediction_rows, metric_rows, parameter_rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='11,23,37,53,71,89,107,127,151,173,197,223,251,281,313,347,383,421,463,509')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--physics-weight', type=float, default=0.001)
    parser.add_argument('--monotonicity-weight', type=float, default=0.02)
    parser.add_argument('--initial-weight', type=float, default=0.05)
    parser.add_argument('--models', default=','.join(MODEL_NAMES))
    parser.add_argument('--run-name', default='primary_training_only_20seeds')
    parser.add_argument('--train-observation-fraction', type=float, default=1.0)
    parser.add_argument('--internal-validation-fraction', type=float, default=0.0)
    parser.add_argument('--time-scaling', choices=['training_only', 'full_series'], default='training_only')
    parser.add_argument('--amplitude-scaling', choices=['training_only', 'full_series', 'per_curve_training'], default='training_only')
    parser.add_argument('--collocation-sampling', choices=['uniform', 'gate_aware'], default='uniform')
    parser.add_argument('--collocation-count', type=int, default=18)
    parser.add_argument('--collocation-max-tau', type=float, default=3.0)
    parser.add_argument('--gate-slope', type=float, default=200.0)
    parser.add_argument('--onset-max-tau', type=float, default=3.0)
    parser.add_argument('--activation-detection-percent', type=float, default=0.02)
    parser.add_argument('--activation-margin-tau', type=float, default=0.1)
    parser.add_argument('--embedding-dim', type=int, default=8)
    parser.add_argument('--no-embedding', action='store_true')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=1e-06)
    parser.add_argument('--history-interval', type=int, default=50)
    parser.add_argument('--exclude-source-record-id', default='')
    parser.add_argument('--holdout-source-record-id', default='')
    parser.add_argument('--drop-sulfate-ratio', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    model_names = [value.strip() for value in args.models.split(',') if value.strip()]
    unknown = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f'unknown models: {unknown}')
    run_out = OUT / args.run_name
    run_out.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(args)
    all_predictions: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    all_parameters: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    all_residuals: list[dict[str, object]] = []
    start = time.perf_counter()
    for model_name in model_names:
        for seed in seeds:
            print(f'training model={model_name} seed={seed}', flush=True)
            model, history = train_one(model_name, seed, prepared, args)
            predictions, metrics, parameters = evaluate(model, model_name, seed, prepared)
            all_predictions.extend(predictions)
            all_metrics.extend(metrics)
            all_parameters.extend(parameters)
            all_history.extend(({'model': model_name, 'seed': seed, **row} for row in history))
            all_residuals.extend(residual_diagnostics(model, model_name, seed, prepared, args))
            checkpoint = run_out / 'checkpoints'
            checkpoint.mkdir(exist_ok=True)
            torch.save(model.state_dict(), checkpoint / f'{model_name}_{seed}.pt')
            for filename, rows in [('neural_predictions.csv', all_predictions), ('neural_seed_metrics.csv', all_metrics), ('neural_inferred_parameters.csv', all_parameters), ('neural_training_history.csv', all_history), ('neural_residual_diagnostics.csv', all_residuals)]:
                if rows:
                    temp = run_out / (filename + '.tmp')
                    pd.DataFrame(rows).to_csv(temp, index=False)
                    temp.replace(run_out / filename)
            (run_out / 'run_progress.json').write_text(json.dumps({'status': 'running', 'completed_models_seeds': len(all_metrics) // len(set(prepared['evaluation_split'])), 'last_model': model_name, 'last_seed': seed, 'configuration': vars(args)}, indent=2))
    predictions_frame = pd.DataFrame(all_predictions)
    metrics_frame = pd.DataFrame(all_metrics)
    parameters_frame = pd.DataFrame(all_parameters)
    history_frame = pd.DataFrame(all_history)
    predictions_frame.to_csv(run_out / 'neural_predictions.csv', index=False)
    metrics_frame.to_csv(run_out / 'neural_seed_metrics.csv', index=False)
    parameters_frame.to_csv(run_out / 'neural_inferred_parameters.csv', index=False)
    history_frame.to_csv(run_out / 'neural_training_history.csv', index=False)
    if all_residuals:
        pd.DataFrame(all_residuals).to_csv(run_out / 'neural_residual_diagnostics.csv', index=False)
    summary_rows = []
    for (model_name, split_name), group in metrics_frame.groupby(['model', 'split']):
        summary_rows.append({'model': model_name, 'split': split_name, 'seeds': group['seed'].nunique(), 'mae_mean_percent': float(group['mae_percent'].mean()), 'mae_std_percent': float(group['mae_percent'].std(ddof=1)), 'mae_min_percent': float(group['mae_percent'].min()), 'mae_max_percent': float(group['mae_percent'].max()), 'rmse_mean_percent': float(group['rmse_percent'].mean()), 'rmse_std_percent': float(group['rmse_percent'].std(ddof=1)), 'rmse_min_percent': float(group['rmse_percent'].min()), 'rmse_max_percent': float(group['rmse_percent'].max()), 'r2_mean': float(group['r2'].mean()), 'r2_std': float(group['r2'].std(ddof=1)), 'mean_curve_mae_mean_percent': float(group['mean_curve_mae_percent'].mean())})
    summary_frame = pd.DataFrame(summary_rows).sort_values(['split', 'rmse_mean_percent'])
    summary_frame.to_csv(run_out / 'neural_summary_metrics.csv', index=False)
    metadata = vars(args) | {'software_versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}, 'avrami_clock': 'anchored logistic integral, fractional-power floor 1e-12', 'models': model_names, 'seeds': seeds, 'runtime_seconds': time.perf_counter() - start, 'effective_train_observations': int(np.sum(prepared['train_mask'])), 'curve_count': int(prepared['curve_frame'].shape[0]), 'observation_count': int(prepared['data'].shape[0]), 'target_scale_percent': prepared['y_scale'], 'onset_upper_min': float(np.min(prepared['onset_upper'])), 'onset_upper_median': float(np.median(prepared['onset_upper'])), 'onset_upper_max': float(np.max(prepared['onset_upper'])), 'features': prepared['features'], 'trainable_parameters': {model_name: int(sum((p.numel() for p in ParametricStateNet(int(prepared['curve_frame'].shape[0]), int(prepared['curve_descriptors'].shape[1]), 0 if args.no_embedding else args.embedding_dim, prepared['onset_upper'], args.width, args.depth).parameters() if p.requires_grad))) for model_name in model_names}, 'feature_medians': prepared['feature_medians'], 'feature_means': prepared['feature_means'], 'feature_stds': prepared['feature_stds'], 'summary': summary_frame.to_dict(orient='records')}
    (run_out / 'neural_model_run_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (run_out / 'run_progress.json').write_text(json.dumps({'status': 'complete', 'configuration': vars(args)}, indent=2))
    print(summary_frame.to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
