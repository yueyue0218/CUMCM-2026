import pytest

from src.q2.main import main, print_summary, run_experiment


def test_smoke_run_is_deterministic_and_reports_both_vertical_sides(capsys):
    first = run_experiment(prior_draws=500, coarse_spacing_m=600.0,
                           refinement_steps_m=(100.0,), direction_bins=360,
                           circle_vertices=36)
    second = run_experiment(prior_draws=500, coarse_spacing_m=600.0,
                            refinement_steps_m=(100.0,), direction_bins=360,
                            circle_vertices=36)
    assert first == second
    print_summary(first)
    output = capsys.readouterr().out
    assert "pure Bayesian" in output
    assert "pure minimax" in output
    assert "robust-envelope hybrid" in output
    assert "vertical +n" in output and "vertical -n" in output


def test_smoke_run_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        run_experiment(prior_draws=0)
    with pytest.raises(ValueError):
        run_experiment(rho=-0.1)


def test_cli_requires_explicit_quick_mode():
    with pytest.raises(SystemExit):
        main([])


def test_main_hybrid_uses_numerical_proxy_envelope(monkeypatch):
    import src.q2.main as module

    calls = []
    original = module.select_robust_envelope_hybrid

    def wrapped(scores, **kwargs):
        calls.append(kwargs.get("use_outer_envelope", False))
        return original(scores, **kwargs)

    monkeypatch.setattr(module, "select_robust_envelope_hybrid", wrapped)
    run_experiment(prior_draws=500, coarse_spacing_m=600.0,
                   refinement_steps_m=(100.0,), direction_bins=360,
                   circle_vertices=36)
    assert calls == [False, False]
