from types import SimpleNamespace

import src.q2.strategy_study as study


def _score(q=(1.0, 2.0), psi=3.0, proxy=4.0, outer=5.0):
    return SimpleNamespace(q=q, bayes=SimpleNamespace(psi_d_m=psi),
                           robust=SimpleNamespace(u_proxy_m=proxy, u_bar_m=outer,
                                                   movement_m=6.0, in_c_rec_certified=True))


def test_rho_sensitivity_reuses_same_scores_and_order(monkeypatch):
    calls = []
    def selector(scores, **kwargs):
        calls.append((id(scores), kwargs["rho"]))
        return scores[0]
    monkeypatch.setattr(study, "select_robust_envelope_hybrid", selector)
    scores = (_score(),)
    result = study.select_rho_sensitivity(scores)
    assert tuple(result) == study.RHO_VALUES
    assert {call[0] for call in calls} == {id(scores)}
    assert [call[1] for call in calls] == list(study.RHO_VALUES)


def test_strategy_row_schema():
    assert set(study._row("hybrid", _score())) == {
        "strategy", "q_x", "q_y", "psi_d_m", "u_proxy_m", "u_bar_m",
        "movement_m", "in_c_rec_certified",
    }
