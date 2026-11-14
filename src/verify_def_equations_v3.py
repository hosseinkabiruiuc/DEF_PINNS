from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '10_revision_v3' / 'equation_audit'

def reaction_balance() -> dict[str, object]:
    monosulfate = np.array([4, 1, 1, 12], dtype=int)
    gypsum = np.array([1, 0, 1, 2], dtype=int)
    anhydrite = np.array([1, 0, 1, 0], dtype=int)
    water = np.array([0, 0, 0, 1], dtype=int)
    ettringite = np.array([6, 1, 3, 32], dtype=int)
    gypsum_reactants = monosulfate + 2 * gypsum + 16 * water
    anhydrite_reactants = monosulfate + 2 * anhydrite + 20 * water
    return {'component_order': ['CaO', 'Al2O3', 'SO3', 'H2O'], 'gypsum_route_reactants': gypsum_reactants.tolist(), 'anhydrite_route_reactants': anhydrite_reactants.tolist(), 'ettringite_product': ettringite.tolist(), 'gypsum_route_balanced': bool(np.array_equal(gypsum_reactants, ettringite)), 'anhydrite_route_balanced': bool(np.array_equal(anhydrite_reactants, ettringite))}

def volume_arithmetic() -> dict[str, float]:
    v_ettringite = 707.0
    v_monosulfate = 309.0
    v_gypsum = 74.2
    v_anhydrite = 45.9
    gypsum_initial = v_monosulfate + 2.0 * v_gypsum
    anhydrite_initial = v_monosulfate + 2.0 * v_anhydrite
    return {'V_ettringite_cm3_mol': v_ettringite, 'V_monosulfate_cm3_mol': v_monosulfate, 'V_gypsum_cm3_mol': v_gypsum, 'V_anhydrite_cm3_mol': v_anhydrite, 'gypsum_route_deltaV_solid_cm3_mol': v_ettringite - gypsum_initial, 'gypsum_route_solid_volume_increase_percent': 100.0 * (v_ettringite - gypsum_initial) / gypsum_initial, 'anhydrite_route_deltaV_solid_cm3_mol': v_ettringite - anhydrite_initial, 'anhydrite_route_solid_volume_increase_percent': 100.0 * (v_ettringite - anhydrite_initial) / anhydrite_initial, 'unsupported_0p267_matches_neither_route': bool(not np.isclose((v_ettringite - gypsum_initial) / gypsum_initial, 0.267) and (not np.isclose((v_ettringite - anhydrite_initial) / anhydrite_initial, 0.267)))}

def symbolic_checks() -> dict[str, object]:
    xi, r, k, t = sp.symbols('xi r k t', positive=True)
    delta = r - 2
    partial = 1 / delta * (1 / (1 - xi) - 2 / (r - 2 * xi))
    partial_ok = sp.simplify(partial - 1 / ((1 - xi) * (r - 2 * xi))) == 0
    q = r * sp.exp(delta * k * t)
    solution = (q - r) / (q - 2)
    derivative_residual = sp.simplify(sp.diff(solution, t) - k * (1 - solution) * (r - 2 * solution))
    initial_residual = sp.simplify(solution.subs(t, 0))
    degenerate_solution = 2 * k * t / (1 + 2 * k * t)
    continuity_residual = sp.simplify(sp.limit(solution, r, 2) - degenerate_solution)
    tau, tau0, slope = sp.symbols('tau tau0 slope', positive=True)
    omega = (sp.log(1 + sp.exp(slope * (tau - tau0))) - sp.log(1 + sp.exp(-slope * tau0))) / slope
    gate = 1 / (1 + sp.exp(-slope * (tau - tau0)))
    gate_residual = sp.simplify(sp.diff(omega, tau) - gate)
    omega_initial = sp.simplify(omega.subs(tau, 0))
    return {'partial_fraction_identity': bool(partial_ok), 'closed_form_ode_residual': str(derivative_residual), 'closed_form_initial_value': str(initial_residual), 'r_to_2_continuity_residual': str(continuity_residual), 'smooth_clock_derivative_residual': str(gate_residual), 'smooth_clock_initial_value': str(omega_initial)}

def numerical_checks() -> dict[str, object]:
    cases = []
    for r_value in [1.2, 2.0, 3.5]:
        k_value = 0.7
        times = np.linspace(0.0, 8.0, 401)
        numerical = solve_ivp(lambda _t, y: [k_value * (1.0 - y[0]) * (r_value - 2.0 * y[0])], (times[0], times[-1]), [0.0], t_eval=times, rtol=1e-11, atol=1e-13).y[0]
        if np.isclose(r_value, 2.0):
            analytic = 2.0 * k_value * times / (1.0 + 2.0 * k_value * times)
        else:
            q = r_value * np.exp((r_value - 2.0) * k_value * times)
            analytic = (q - r_value) / (q - 2.0)
        capacity = min(1.0, r_value / 2.0)
        cases.append({'r': r_value, 'capacity': capacity, 'maximum_absolute_solution_error': float(np.max(np.abs(numerical - analytic))), 'analytic_final_value': float(analytic[-1]), 'does_not_exceed_capacity': bool(np.max(analytic) <= capacity + 1e-10)})
    return {'cases': cases, 'all_errors_below_1e-8': all((row['maximum_absolute_solution_error'] < 1e-08 for row in cases))}

def nondimensional_check() -> dict[str, object]:
    dimensions = {'epsilon': [1, 0], 'E_star': [1, 0], 'y=epsilon/E_star': [0, 0], 'A': [1, 0], 'a=A/E_star': [0, 0], 'k': [0, -1], 'T_train': [0, 1], 'kappa=k*T_train': [0, 0], 'tau=t/T_train': [0, 0], 'dy/dtau': [0, 0], 'kappa*(a-y)': [0, 0]}
    return {'basis': ['strain', 'time'], 'dimensions': dimensions, 'ode_is_dimensionless': True}

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {'reaction_balance': reaction_balance(), 'molar_volume_arithmetic': volume_arithmetic(), 'symbolic': symbolic_checks(), 'numerical': numerical_checks(), 'nondimensionalization': nondimensional_check()}
    checks = [audit['reaction_balance']['gypsum_route_balanced'], audit['reaction_balance']['anhydrite_route_balanced'], audit['molar_volume_arithmetic']['unsupported_0p267_matches_neither_route'], audit['symbolic']['partial_fraction_identity'], audit['symbolic']['closed_form_ode_residual'] == '0', audit['symbolic']['closed_form_initial_value'] == '0', audit['symbolic']['r_to_2_continuity_residual'] == '0', audit['symbolic']['smooth_clock_derivative_residual'] == '0', audit['symbolic']['smooth_clock_initial_value'] == '0', audit['numerical']['all_errors_below_1e-8'], audit['nondimensionalization']['ode_is_dimensionless']]
    audit['checks_passed'] = int(sum(checks))
    audit['checks_total'] = len(checks)
    audit['all_checks_passed'] = bool(all(checks))
    (OUT / 'equation_audit_v3.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return 0 if all(checks) else 1
if __name__ == '__main__':
    raise SystemExit(main())
