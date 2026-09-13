from types import SimpleNamespace

import pytest

import src.q2.convergence as convergence


def _fake_summary(draws):
    def score(q, psi, proxy, outer):
        return SimpleNamespace(
            q=q,
            bayes=SimpleNamespace(psi_d_m=psi),
            robust=SimpleNamespace(
                u_proxy_m=proxy, u_bar_m=outer,
                movement_m=100.0, in_c_rec_certified=True,
            ),
        )

    return SimpleNamespace(
        posterior=SimpleNamespace(
            retained_draws=draws // 2,
            effective_sample_size=float(draws // 3),
            acceptance_rate=0.5,
        ),
        pure_bayesian=score((1.0, 2.0), 1.0, 2.0, 3.0),
        pure_minimax=score((3.0, 4.0), 2.0, 1.0, 2.0),
        hybrid=score((5.0, 6.0), 1.5, 1.5, 2.5),
    )


def test_runner_small_sizes_schema_and_determinism(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(**kwargs):
        calls.append(kwargs)
        return _fake_summary(kwargs["prior_draws"])

    monkeypatch.setattr(convergence, "run_experiment", fake_run_experiment)
    rows = convergence.run_convergence((10, 20), output_dir=tmp_path)
    assert len(rows) == 6
    assert {row.strategy for row in rows} == {"pure_bayesian", "pure_minimax", "hybrid"}
    assert len(calls) == 2
    assert all(call["direction_bins"] == 180 for call in calls)
    assert all(row.posterior_ess > 0 for row in rows)
    assert (tmp_path / "posterior_convergence.csv").exists()
    assert (tmp_path / "posterior_convergence.md").exists()


def test_invalid_sample_sizes_fail_fast(tmp_path):
    with pytest.raises(ValueError):
        convergence.run_convergence((0,), output_dir=tmp_path)
    with pytest.raises(ValueError):
        convergence.run_convergence((10, 10), output_dir=tmp_path)


def _row(size, strategy, dq, dpsi, du):
    return convergence.ConvergenceRow(
        prior_draws=size, runtime_s=1.0, posterior_retained_draws=10,
        posterior_ess=8.0, acceptance_rate=0.1, strategy=strategy,
        q_x=0.0, q_y=0.0, psi_d_m=1.0, u_proxy_m=2.0, u_bar_m=3.0,
        movement_m=4.0, in_c_rec_certified=True,
        q_displacement_m=dq, psi_abs_change_m=dpsi,
        psi_relative_change=dpsi, u_proxy_abs_change_m=du,
        u_proxy_relative_change=du, u_bar_abs_change_m=0.0,
        u_bar_relative_change=0.0,
    )


def test_stability_gate_passes_and_fails_explicitly():
    passing = ([_row(300000, strategy, None, None, None)
                for strategy in convergence.STRATEGIES]
               + [_row(600000, strategy, 20.0, 0.05, 0.05)
                  for strategy in convergence.STRATEGIES])
    assert convergence.stability_gate(
        passing, previous_size=300000, current_size=600000)["passed"]
    failing = ([_row(300000, strategy, None, None, None)
                for strategy in convergence.STRATEGIES]
               + [_row(600000, strategy, 100.0, 0.2, 0.2)
                  for strategy in convergence.STRATEGIES])
    assert not convergence.stability_gate(
        failing, previous_size=300000, current_size=600000)["passed"]


def test_symmetry_aware_displacement_is_explicit():
    assert convergence.symmetry_aware_displacement((0.0, -480.0), (0.0, 480.0)) == 960.0
    assert convergence.symmetry_aware_displacement(
        (0.0, -480.0), (0.0, 480.0), mode="x_reflection") == 0.0
    assert convergence.symmetry_aware_displacement(
        (28.28, -571.72), (0.0, 560.0), mode="x_reflection") < 31.0
    with pytest.raises(ValueError):
        convergence.symmetry_aware_displacement((0.0, 0.0), (1.0, 1.0), mode="auto")


def test_gated_runner_does_not_enter_numerical_checks_when_gate_fails(monkeypatch, tmp_path):
    rows = ([_row(300000, strategy, None, None, None) for strategy in convergence.STRATEGIES]
            + [_row(600000, strategy, 100.0, 0.2, 0.2) for strategy in convergence.STRATEGIES])
    monkeypatch.setattr(convergence, "run_convergence", lambda *args, **kwargs: rows)
    called = []
    monkeypatch.setattr(convergence, "run_numerical_convergence", lambda **kwargs: called.append(kwargs))
    result = convergence.run_gated(existing_output_dir=tmp_path, added_sizes=(300000, 600000, 1000000))
    assert result["gate"]["passed"] is False
    assert called == []
