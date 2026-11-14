from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '.'
REV = PROJECT / 'results'
OUT = REV / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
C100 = REV / 'c100_revision'
OBS = pd.read_csv(PROJECT / 'data/modeling' / 'def_kinetics_observations_v2.csv')
CURVES = pd.read_csv(PROJECT / 'data/modeling' / 'def_kinetics_curves_v2.csv')
COLORS = {'pinn_shifted_first_order': '#7562A6', 'data_only': '#5D84B4', 'pinn_avrami': '#C28C5A', 'shape_regularized': '#73A57B', 'shifted_first_order': '#4D9B83', 'larive': '#A7799B', 'nlme_dfo': '#B8874A', 'persistence': '#727D8A', 'linear_last5': '#9AA4AE', 'first_order': '#9B83B7', 'shifted_avrami': '#79A8A1', 'gompertz': '#648FB0'}
SOURCE_COLORS = {'Aubert et al. 2013': '#5D84B4', 'Brunetaud et al. 2007': '#8A74B0', 'Kawabata et al. 2023': '#4D9B83', 'Martin et al. 2023': '#C28C5A', 'Tiecher et al. 2021': '#A7799B'}
SOURCE_MARKERS = {key: marker for key, marker in zip(SOURCE_COLORS, ['o', 'P', 'D', '^', 's'])}
MODEL_MARKERS = {'first_order': 'o', 'shifted_first_order': 's', 'shifted_avrami': 'D', 'gompertz': '^', 'larive': 'P'}
LABELS = {'pinn_shifted_first_order': 'Delayed first-order PINNs', 'data_only': 'Data-only NN', 'pinn_avrami': 'Avrami-residual PINNs', 'shape_regularized': 'Shape + initial condition', 'shifted_first_order': 'Delayed first order', 'larive': 'Larive', 'nlme_dfo': 'Partially pooled DFO', 'persistence': 'Persistence', 'linear_last5': 'Last-five linear', 'first_order': 'First order', 'shifted_avrami': 'Delayed Avrami', 'gompertz': 'Gompertz'}

def setup() -> None:
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.0, 'axes.titlesize': 8.6, 'axes.labelsize': 8.0, 'axes.titlepad': 6.0, 'axes.labelpad': 4.0, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 7.0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8, 'axes.edgecolor': '#252B33', 'axes.grid': False, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3.4, 'ytick.major.size': 3.4, 'xtick.major.width': 0.75, 'ytick.major.width': 0.75, 'xtick.minor.size': 2.2, 'ytick.minor.size': 2.2, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6, 'xtick.color': '#252B33', 'ytick.color': '#252B33', 'lines.linewidth': 1.25, 'lines.markersize': 4.0, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})

def panel_labels(axes) -> None:
    for label, ax in zip('abcdefghijklmnopqrstuvwxyz', np.ravel(axes)):
        ax.text(-0.17, 1.105, label, transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='top')

def polish_axis(ax, *, left: bool=True, right: bool=False, bottom: bool=True) -> None:
    ax.tick_params(axis='x', which='major', bottom=bottom, top=False, length=3.4, width=0.75, pad=2.8)
    ax.tick_params(axis='x', which='minor', bottom=bottom, top=False, length=2.2, width=0.6)
    ax.tick_params(axis='y', which='major', left=left, right=right, length=3.4, width=0.75, pad=2.8)
    ax.tick_params(axis='y', which='minor', left=left, right=right, length=2.2, width=0.6)
    ax.title.set_fontsize(8.6)
    ax.xaxis.label.set_fontsize(8.0)
    ax.yaxis.label.set_fontsize(8.0)

def polish_axes(axes) -> None:
    for ax in np.ravel(axes):
        polish_axis(ax)

def assert_uniform_panels(fig, axes, *, equal_heights: bool=True, tolerance: float=0.006) -> None:
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in np.ravel(axes)]
    widths = np.array([box.width for box in boxes])
    heights = np.array([box.height for box in boxes])
    if float(np.ptp(widths)) > tolerance:
        raise RuntimeError(f'panel width mismatch: {widths.tolist()}')
    if equal_heights and float(np.ptp(heights)) > tolerance:
        raise RuntimeError(f'panel height mismatch: {heights.tolist()}')

def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=600, bbox_inches='tight', pad_inches=0.04)
    export = ROOT / 'outputs' / 'figures'
    export.mkdir(exist_ok=True)
    fig.savefig(export / Path(name).with_suffix('.tif'), dpi=1200, bbox_inches='tight', pad_inches=0.04, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)

def figure1() -> None:
    ordered = OBS.sort_values(['curve_id', 'time_days'])
    curve = ordered.groupby('curve_id', as_index=False).agg(source_short=('source_short', 'first'), terminal=('expansion_percent', 'last'))
    source_order = [source for source in SOURCE_COLORS if source in set(curve['source_short'])]
    source_labels = [f"{source.replace(' et al. ', ' ')} (n={int(curve['source_short'].eq(source).sum())})" for source in source_order]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.35), constrained_layout=False, gridspec_kw={'height_ratios': [1.15, 1.0]})
    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.15, hspace=0.42)
    ax = axes[0]
    values = [curve.loc[curve['source_short'].eq(source), 'terminal'].to_numpy(float) for source in source_order]
    boxes = ax.boxplot(values, vert=False, positions=np.arange(len(source_order)), widths=0.52, patch_artist=True, showfliers=False, medianprops={'color': '#111827', 'lw': 1.0}, whiskerprops={'color': '#6B7280', 'lw': 0.75}, capprops={'color': '#6B7280', 'lw': 0.75}, boxprops={'edgecolor': '#6B7280', 'lw': 0.75})
    rng = np.random.default_rng(17)
    for position, source, source_values, box in zip(np.arange(len(source_order)), source_order, values, boxes['boxes']):
        box.set_facecolor(SOURCE_COLORS[source])
        box.set_alpha(0.22)
        ax.scatter(source_values, position + rng.normal(0, 0.055, len(source_values)), s=18, marker=SOURCE_MARKERS[source], color=SOURCE_COLORS[source], alpha=0.8, edgecolor='white', linewidth=0.3, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Terminal expansion (% strain)')
    ax.set_yticks(np.arange(len(source_order)), source_labels)
    ax.invert_yaxis()
    ax.set_title('Terminal response by source')
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax = axes[1]
    class_specs = [('Negligible', -np.inf, 0.04, '#5D84B4'), ('Moderate', 0.04, 0.6, '#7562A6'), ('Large', 0.6, np.inf, '#4D9B83')]
    counts = np.zeros((len(source_order), len(class_specs)), dtype=int)
    for source_index, source in enumerate(source_order):
        terminals = curve.loc[curve['source_short'].eq(source), 'terminal'].to_numpy(float)
        for class_index, (_, lower, upper, _) in enumerate(class_specs):
            counts[source_index, class_index] = int(np.sum((terminals >= lower) & (terminals < upper)))
    percentages = 100 * counts / counts.sum(axis=1, keepdims=True)
    left = np.zeros(len(source_order))
    positions = np.arange(len(source_order))
    for class_index, (label, _, _, color) in enumerate(class_specs):
        widths = percentages[:, class_index]
        ax.barh(positions, widths, left=left, height=0.56, color=color, edgecolor='#EDF0F2', linewidth=0.8, label=label, hatch=['', '///', '...'][class_index])
        for position, start, width, count in zip(positions, left, widths, counts[:, class_index]):
            if width >= 12:
                ax.text(start + width / 2, position, f'n={count}', ha='center', va='center', fontsize=7.1, color='white', fontweight='semibold', bbox={'facecolor': color, 'edgecolor': 'none', 'pad': 0.8})
        left += widths
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Share of curves (%)')
    ax.set_yticks(positions, [source.replace(' et al. ', ' ') for source in source_order])
    ax.invert_yaxis()
    ax.set_title('Response-class composition by source')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.015), frameon=False)
    polish_axes(axes)
    panel_labels(axes)
    assert_uniform_panels(fig, axes, equal_heights=False)
    save(fig, 'Figure_1_dataset_heterogeneity.png')

def figure2() -> None:
    metrics = pd.read_csv(REV / 'direct_kinetic_fits' / 'c100_primary_with_larive' / 'direct_fit_per_curve_metrics.csv')
    metrics = metrics[metrics['split'].eq('terminal_test')]
    models = ['first_order', 'shifted_first_order', 'shifted_avrami', 'gompertz', 'larive']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.88, wspace=0.28)
    ax = axes[0]
    line_styles = {'first_order': '-', 'shifted_first_order': '--', 'shifted_avrami': '-.', 'gompertz': (0, (3, 1, 1, 1)), 'larive': (0, (1, 1))}
    for model in models:
        values = np.sort(metrics.loc[metrics['model'].eq(model), 'rmse_percent'].to_numpy(float))
        values = np.maximum(values, 1e-06)
        exceedance = 100 * np.arange(len(values), 0, -1) / len(values)
        ax.step(values, exceedance, where='post', lw=1.35, ls=line_styles[model], color=COLORS[model], label=LABELS[model])
        indices = np.unique(np.linspace(0, len(values) - 1, 10, dtype=int))
        ax.scatter(values[indices], exceedance[indices], s=14, marker=MODEL_MARKERS[model], color=COLORS[model], edgecolor='white', linewidth=0.25, zorder=3)
    ax.set_xscale('log')
    ax.set_xlim(left=max(metrics['rmse_percent'].min() * 0.72, 1e-05))
    ax.set_ylim(0, 103)
    ax.set_xlabel('Curve terminal RMSE (% strain)')
    ax.set_ylabel('Curves exceeding error (%)')
    ax.set_title('Terminal-error exceedance')
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, ncol=1, loc='upper right', borderaxespad=0.35, handlelength=1.8, handletextpad=0.45, labelspacing=0.32)
    ax = axes[1]
    ident = pd.read_csv(REV / 'direct_kinetic_fits' / 'c100_primary_with_larive' / 'direct_fit_parameters.csv')
    ident_models = ['first_order', 'shifted_first_order', 'shifted_avrami', 'gompertz', 'larive']
    rng = np.random.default_rng(31)
    positions = np.arange(len(ident_models))
    for position, model in zip(positions, ident_models):
        group = ident[ident.model.eq(model)].copy()
        unresolved = group.jacobian_information_condition_number.to_numpy(float) > 1e+16
        values = np.clip(np.sqrt(group.jacobian_information_condition_number.to_numpy(float)), 1.0, 10000000000000.0)
        jitter = rng.uniform(-0.18, 0.18, len(group))
        hit = group.bound_hit_count.to_numpy(int) > 0
        marker = MODEL_MARKERS[model]
        ax.scatter(position + jitter[~hit], values[~hit], s=20, marker=marker, facecolor='white', edgecolor=COLORS[model], linewidth=0.8, alpha=0.76, zorder=2)
        ax.scatter(position + jitter[hit], values[hit], s=21, marker=marker, color=COLORS[model], edgecolor='white', linewidth=0.3, alpha=0.76, zorder=3)
        ax.scatter(position + jitter[unresolved], values[unresolved], s=28, marker='x', color='#20242A', linewidth=0.8, zorder=5)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(position, q1, q3, color='#2F3742', lw=2.7, zorder=4)
        ax.hlines(median, position - 0.22, position + 0.22, color='#111827', lw=1.2, zorder=5)
    ax.set_yscale('log')
    ax.set_xticks(positions, ['First\norder', 'Delayed\nfirst order', 'Delayed\nAvrami', 'Gompertz', 'Larive'])
    ax.set_ylabel('Jacobian condition number, $\\mathrm{cond}(J)$')
    ax.set_title('Direct-fit conditioning')
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(handles=[mpl.lines.Line2D([], [], marker='o', linestyle='none', mfc='white', mec='#59616C', ms=4.2, label='No bound contact'), mpl.lines.Line2D([], [], marker='o', linestyle='none', mfc='#59616C', mec='white', ms=4.5, label='Bound contact'), mpl.lines.Line2D([], [], marker='x', linestyle='none', color='#20242A', ms=4.5, label='$\\mathrm{cond}(J)>10^{8}$')], frameon=False, loc='upper left', handletextpad=0.35, labelspacing=0.35)
    polish_axes(axes)
    panel_labels(axes)
    assert_uniform_panels(fig, axes)
    save(fig, 'Figure_2_direct_model_error_tails.png')

def load_summary(run: str, model: str) -> pd.Series:
    frame = pd.read_csv(REV / 'neural_models' / run / 'neural_summary_metrics.csv')
    return frame[frame.model.eq(model) & frame.split.eq('terminal_test')].iloc[0]

def figure3() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.15), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.885, bottom=0.105, top=0.95, wspace=0.38, hspace=0.43)
    for panel in np.ravel(axes):
        panel.set_box_aspect(1)
    primary = pd.read_csv(REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_seed_metrics.csv')
    primary = primary[primary.split.eq('terminal_test')]
    ax = axes[0, 0]
    models = ['data_only', 'pinn_shifted_first_order', 'pinn_avrami']
    for i, m in enumerate(models):
        g = primary[primary.model.eq(m)]
        violin = ax.violinplot(g.mean_curve_mae_percent, positions=[i], widths=0.62, showmeans=False, showmedians=False, showextrema=False)
        for body in violin['bodies']:
            body.set_facecolor(COLORS[m])
            body.set_edgecolor(COLORS[m])
            body.set_alpha(0.18)
        ax.scatter(np.full(len(g), i) + np.linspace(-0.11, 0.11, len(g)), g.mean_curve_mae_percent, s=14, color=COLORS[m], alpha=0.72, edgecolor='white', linewidth=0.25)
        ax.hlines(g.mean_curve_mae_percent.mean(), i - 0.22, i + 0.22, color='#20242A', lw=1.4)
    ax.set_xticks(range(3), ['Data-only', 'Delayed FO\nPINNs', 'Avrami\nPINNs'])
    ax.set_ylabel('Mean curve MAE (% strain)')
    ax.set_title('Matched-seed forecasts')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax = axes[0, 1]
    cv_runs = [(3e-05, 'reviewer_v5_lambda_cv_0p00003_5seeds'), (0.0001, 'reviewer_v5_lambda_cv_0p0001_5seeds'), (0.0003, 'reviewer_v5_lambda_cv_0p0003_5seeds'), (0.001, 'reviewer_v5_lambda_cv_0p001_5seeds'), (0.003, 'reviewer_v5_lambda_cv_0p003_5seeds'), (0.01, 'reviewer_v5_lambda_cv_0p01_5seeds')]
    cv = []
    for lam, run in cv_runs:
        values = pd.read_csv(REV / 'neural_models' / run / 'neural_seed_metrics.csv')
        values = values[values.model.eq('pinn_shifted_first_order') & values.split.eq('internal_validation')]
        cv.append((lam, values.mean_curve_mae_percent.mean(), values.mean_curve_mae_percent.std(ddof=1)))
    cv = pd.DataFrame(cv, columns=['lambda', 'mean', 'sd'])
    ax.errorbar(cv['lambda'], cv['mean'], yerr=cv['sd'], fmt='o-', ms=4.2, lw=1.25, capsize=2.5, color=COLORS['pinn_shifted_first_order'])
    chosen = cv.loc[cv['mean'].idxmin()]
    ax.scatter(chosen['lambda'], chosen['mean'], s=52, facecolor='white', edgecolor='#20242A', linewidth=1.0, zorder=5)
    ax.scatter(chosen['lambda'], chosen['mean'], s=19, color=COLORS['pinn_shifted_first_order'], zorder=6)
    ax.annotate('selected', xy=(chosen['lambda'], chosen['mean']), xytext=(-8, 42), textcoords='offset points', ha='center', va='bottom', fontsize=7.0, color='#4B5563', arrowprops={'arrowstyle': '-', 'linestyle': '--', 'color': '#727D8A', 'lw': 0.75, 'shrinkA': 3, 'shrinkB': 4})
    ax.set_xscale('log')
    ax.set_xlabel('Physics-loss weight, $\\lambda_{phys}$')
    ax.set_ylabel('Validation MAE (% strain)')
    ax.set_title('Endpoint-blind weight selection')
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax = axes[1, 0]
    paired = pd.read_csv(REV / 'compiled' / 'curve_level_seed_averaged_errors.csv')
    for source, g in paired.groupby('source_short'):
        ax.scatter(g.mae_percent_data_only, g.mae_percent_pinn_shifted_first_order, s=20, color=SOURCE_COLORS[source], marker=SOURCE_MARKERS[source], alpha=0.76, linewidth=0.45, label=source.split()[0])
    limit = 1.04 * max(float(paired.mae_percent_data_only.max()), float(paired.mae_percent_pinn_shifted_first_order.max()))
    ax.plot([0, limit], [0, limit], ls='--', lw=1, color='#727D8A')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('Data-only MAE (% strain)')
    ax.set_ylabel('FO PINNs MAE (% strain)')
    ax.set_title('Paired curve errors')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(frameon=False, ncol=2, loc='upper left', handletextpad=0.25, columnspacing=0.7)
    ax = axes[1, 1]
    trade = pd.read_csv(REV / 'reviewer_diagnostics' / 'review75_compliance_tradeoff.csv').sort_values('lambda_phys')
    positive = trade[trade.lambda_phys.gt(0)]
    ax.plot(positive.lambda_phys, positive.median_normalized_residual_observed, 'o-', ms=4.2, lw=1.25, color=COLORS['pinn_shifted_first_order'], label='Normalized residual')
    ax.set_xscale('log')
    ax.set_ylim(0.45, 1.04)
    ax.set_xlabel('Physics-loss weight, $\\lambda_{phys}$')
    ax.set_ylabel('Normalized residual', color=COLORS['pinn_shifted_first_order'])
    ax.spines['left'].set_color(COLORS['pinn_shifted_first_order'])
    ax.tick_params(axis='y', which='both', colors=COLORS['pinn_shifted_first_order'])
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_yticks(np.arange(0.45, 1.041, 0.05), minor=True)
    ax2 = ax.twinx()
    ax2.plot(positive.lambda_phys, positive.terminal_mae_percent, 's-', ms=4.0, lw=1.15, color=COLORS['shape_regularized'], label='Terminal MAE')
    ax2.set_ylim(0.06, 0.111)
    ax2.set_yticks(np.arange(0.06, 0.111, 0.01))
    ax2.set_yticks(np.arange(0.065, 0.111, 0.01), minor=True)
    ax2.set_ylabel('Pooled MAE (% strain)', color=COLORS['shape_regularized'])
    ax2.tick_params(axis='y', which='both', colors=COLORS['shape_regularized'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(COLORS['shape_regularized'])
    selected = positive[np.isclose(positive.lambda_phys, float(chosen['lambda']))].iloc[0]
    ax.scatter(selected.lambda_phys, selected.median_normalized_residual_observed, s=50, facecolor='white', edgecolor='#20242A', linewidth=1.0, zorder=5)
    ax.scatter(selected.lambda_phys, selected.median_normalized_residual_observed, s=18, color=COLORS['pinn_shifted_first_order'], zorder=6)
    ax2.scatter(selected.lambda_phys, selected.terminal_mae_percent, s=48, facecolor='white', edgecolor='#20242A', linewidth=1.0, zorder=5)
    ax2.scatter(selected.lambda_phys, selected.terminal_mae_percent, s=17, marker='s', color=COLORS['shape_regularized'], zorder=6)
    ax.annotate('selected', xy=(selected.lambda_phys, selected.median_normalized_residual_observed), xytext=(0, -25), textcoords='offset points', ha='center', va='top', fontsize=7.0, color='#4B5563', arrowprops={'arrowstyle': '-', 'linestyle': '--', 'color': '#727D8A', 'lw': 0.75, 'shrinkA': 3, 'shrinkB': 4})
    ax.set_title('Compliance-accuracy tradeoff')
    polish_axes(axes)
    polish_axis(ax2, left=False, right=True, bottom=False)
    ax2.set_box_aspect(1)
    panel_labels(axes)
    assert_uniform_panels(fig, axes)
    save(fig, 'Figure_3_neural_benchmark.png')

def figure4() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.15), constrained_layout=False)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.15, top=0.95, wspace=0.28, hspace=0.43)
    for panel in np.ravel(axes):
        panel.set_box_aspect(1)
    cases = [('Aubert2013_Fig2_C20', 'Late activation: Aubert C20'), ('Kawabata2023_Fig5b_550mm', 'Plateau: Kawabata 550 mm'), ('Tiecher2021_Fig2_CP_II-Z', 'Non-monotone: Tiecher CP II-Z')]
    for index, (ax, (curve_id, title)) in enumerate(zip(np.ravel(axes)[:3], cases)):
        curve_panel(ax, curve_id, title, models=None if index < 2 else ['pinn_shifted_first_order'])
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax = axes[1, 1]
    predictions = seed_mean_predictions(pd.read_csv(REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_predictions.csv'))
    predictions = predictions[predictions.model.eq('pinn_shifted_first_order') & predictions.split.eq('terminal_test')]
    for source, g in predictions.groupby('source_short'):
        ax.scatter(g.observed_expansion_percent, g.predicted_expansion_percent, s=20, color=SOURCE_COLORS[source], marker=SOURCE_MARKERS[source], alpha=0.72, linewidth=0.45, label=source.split()[0])
    limit = 1.04 * max(float(predictions.observed_expansion_percent.max()), float(predictions.predicted_expansion_percent.max()))
    ax.plot([0, limit], [0, limit], ls='--', lw=1, color='#727D8A')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('Observed expansion (% strain)')
    ax.set_ylabel('PINNs prediction (% strain)')
    ax.set_title('All held-out observations')
    errors = predictions.predicted_expansion_percent - predictions.observed_expansion_percent
    mae = float(errors.abs().mean())
    centered = predictions.observed_expansion_percent - predictions.observed_expansion_percent.mean()
    r_squared = 1.0 - float((errors ** 2).sum() / (centered ** 2).sum())
    ax.text(0.045, 0.955, f'MAE = {mae:.3f}%\n$R^2$ = {r_squared:.3f}', transform=ax.transAxes, ha='left', va='top', fontsize=7.0, color='#4B5563', linespacing=1.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    handles = [mpl.lines.Line2D([], [], marker='o', linestyle='none', mfc='#172033', mec='#172033', ms=4.5, label='Training observation'), mpl.lines.Line2D([], [], marker='o', linestyle='none', mfc='white', mec='#172033', ms=4.5, label='Held-out observation'), mpl.lines.Line2D([], [], color=COLORS['data_only'], lw=1.4, ls='-.', marker='s', ms=3, label='Data-only'), mpl.lines.Line2D([], [], color=COLORS['pinn_shifted_first_order'], lw=1.4, marker='o', ms=3, label='Delayed FO PINNs'), mpl.lines.Line2D([], [], color=COLORS['shifted_first_order'], lw=1.4, ls='--', marker='^', ms=3, label='Direct DFO'), mpl.lines.Line2D([], [], color=COLORS['persistence'], lw=1.4, ls=':', marker='D', ms=3, label='Persistence')]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.02), frameon=False)
    polish_axes(axes)
    panel_labels(axes)
    assert_uniform_panels(fig, axes)
    save(fig, 'Figure_4_representative_predictions.png')

def curve_panel(ax, curve_id: str, title: str, models: list[str] | None=None) -> None:
    obs = OBS[OBS.curve_id.eq(curve_id)].sort_values('time_days')
    train = obs[obs.split.eq('train')]
    test = obs[obs.split.eq('terminal_test')]
    ax.plot(train.time_days, train.expansion_percent, 'o', ms=3.4, color='#172033', label='Training observations')
    ax.plot(test.time_days, test.expansion_percent, 'o', ms=4, mfc='white', mec='#172033', mew=1, label='Held-out observations')
    neural = seed_mean_predictions(pd.read_csv(REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_predictions.csv'))
    direct = pd.read_csv(REV / 'direct_kinetic_fits' / 'c100_primary_with_larive' / 'direct_fit_predictions.csv')
    base = pd.read_csv(REV / 'compiled' / 'baseline_predictions.csv')
    model_frames = {'pinn_shifted_first_order': neural, 'data_only': neural, 'shifted_first_order': direct, 'persistence': base}
    selected_models = models or ['pinn_shifted_first_order', 'data_only', 'shifted_first_order', 'persistence']
    line_styles = {'pinn_shifted_first_order': '-', 'data_only': '-.', 'shifted_first_order': '--', 'persistence': ':'}
    markers = {'pinn_shifted_first_order': 'o', 'data_only': 's', 'shifted_first_order': '^', 'persistence': 'D'}
    for model in selected_models:
        frame = model_frames[model]
        g = frame[frame.curve_id.eq(curve_id) & frame.model.eq(model)].sort_values('time_days')
        g = g[g.time_days >= train.time_days.max()]
        ax.plot(g.time_days, g.predicted_expansion_percent, lw=1.35, ls=line_styles[model], color=COLORS[model], label=LABELS[model])
        ax.scatter(g.time_days, g.predicted_expansion_percent, s=12, marker=markers[model], color=COLORS[model], edgecolor='white', linewidth=0.25, zorder=3)
    ax.axvline(train.time_days.max(), color='#ABB4BE', ls=':', lw=1)
    ax.set_title(title)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Expansion (% strain)')
    if curve_id == 'Tiecher2021_Fig2_CP_II-Z':
        ax.plot(test.time_days, test.expansion_percent, color='#727D8A', lw=0.85, alpha=0.75, zorder=1)
        minimum = test.loc[test.expansion_percent.idxmin()]
        ax.annotate('Holdout minimum', xy=(minimum.time_days, minimum.expansion_percent), xytext=(-8, -17), textcoords='offset points', ha='right', fontsize=6.8, color='#4B5563', arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': '#727D8A'})
        boundary = train.iloc[-1]
        ax.annotate('Training endpoint', xy=(boundary.time_days, boundary.expansion_percent), xytext=(-2, -28), textcoords='offset points', ha='right', va='top', fontsize=6.8, color='#4B5563', arrowprops={'arrowstyle': '-', 'lw': 0.7, 'color': '#727D8A'})

def seed_mean_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ['observation_id', 'curve_id', 'record_id', 'source_short', 'model', 'split', 'time_days']
    return frame.groupby(keys, as_index=False).agg(observed_expansion_percent=('observed_expansion_percent', 'first'), predicted_expansion_percent=('predicted_expansion_percent', 'mean'))

def figure5() -> None:
    lead = pd.read_csv(C100 / 'error_by_forecast_lead.csv')
    models = ['pinn_shifted_first_order', 'shifted_first_order', 'persistence']
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for ax, scale, xlabel, title in [(axes[0, 0], 'normalized', 'Normalized lead, (t-Tᵢ)/Tᵢ', 'Normalized lead'), (axes[0, 1], 'physical_days', 'Physical lead (days)', 'Physical lead')]:
        for model in models:
            g = lead[lead.scale.eq(scale) & lead.model.eq(model)].sort_values('bin_mid')
            ax.plot(g.bin_mid, g.mae_percent, 'o-', ms=3.5, lw=1.35, color=COLORS[model], label=LABELS[model])
        if scale == 'physical_days':
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('MAE (% strain)')
        ax.set_title(title)
        ax.grid(True)
    curve_panel(axes[1, 0], 'Aubert2013_Fig2_C20', 'Late activation: Aubert C20')
    curve_panel(axes[1, 1], 'Tiecher2021_Fig2_CP_II-Z', 'Measured reversal: Tiecher CP II-Z')
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], lw=1.5, label=LABELS[m]) for m in models]
    fig.legend(handles=handles, ncol=3, loc='outside lower center', frameon=False)
    panel_labels(axes)
    save(fig, 'Figure_5_lead_time_and_failures.png')

def figure6() -> None:
    p = pd.read_csv(C100 / 'identifiable_parameter_summary_by_curve.csv')
    p = p[p.model.eq('pinn_shifted_first_order')]
    sources = list(SOURCE_COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    specs = [('amplitude_percent', 'Expansion capacity, εcap (% strain)', False), ('rate_per_day', 'Rate, k (d⁻¹)', True), ('rate_normalized', 'Dimensionless rate, κ = kTᵢ', True), ('threshold_fraction', 'Threshold fraction, ξc/ξ∞', False)]
    rng = np.random.default_rng(9)
    for ax, (column, ylabel, log) in zip(np.ravel(axes), specs):
        for i, source in enumerate(sources):
            g = p[p.source_short.eq(source)][column].dropna().to_numpy()
            ax.scatter(i + rng.normal(0, 0.045, len(g)), g, s=15, color=SOURCE_COLORS[source], alpha=0.72, edgecolor='white', linewidth=0.3)
            if len(g):
                ax.hlines(np.median(g), i - 0.19, i + 0.19, color='#1F252B', lw=1.4)
        if log:
            ax.set_yscale('log')
        ax.set_xticks(range(len(sources)), [s.split()[0] for s in sources], rotation=18, ha='right')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y')
    panel_labels(axes)
    save(fig, 'Figure_6_identifiable_parameters.png')

def main() -> None:
    setup()
    figure1()
    figure2()
    figure3()
    figure4()
    print('created 4 manuscript figures')
if __name__ == '__main__':
    main()
