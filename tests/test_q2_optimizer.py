import math
import unittest
from unittest.mock import patch

from src.q2.model import (
    BayesianEvaluation,
    BearingErrorAtom,
    FirstDirectionObservation,
    FirstState,
    JointSample,
    Q2Config,
    RobustEvaluation,
)
from src.q2.optimizer import (
    CandidateScore,
    coarse_grid_candidates,
    minimax_baseline,
    score_candidates,
    select_pure_bayesian,
    select_pure_minimax,
    select_robust_envelope_hybrid,
)


class CoarseGridCandidateTests(unittest.TestCase):
    def make_state(
        self,
        outer_region: tuple[tuple[float, float], ...] = (
            (-100.0, -100.0),
            (100.0, -100.0),
            (100.0, 100.0),
            (-100.0, 100.0),
        ),
    ) -> FirstState:
        return FirstState(
            observation=FirstDirectionObservation((0.0, 0.0), 0.0),
            exact_support_label="optimizer test support",
            outer_region=outer_region,
            joint_samples=(JointSample((50.0, 0.0), 1000.0, 1.0),),
        )

    def test_rejects_invalid_spacing(self) -> None:
        state = self.make_state()
        for spacing in (0.0, -1.0, math.inf, math.nan, True):
            with self.subTest(spacing=spacing):
                with self.assertRaises(ValueError):
                    coarse_grid_candidates(state, spacing_m=spacing)  # type: ignore[arg-type]

    def test_all_returned_points_are_in_c_poss_proxy(self) -> None:
        from src.q2.model import evaluate_candidate_domains

        state = self.make_state()
        candidates = coarse_grid_candidates(
            state,
            spacing_m=500.0,
            config=Q2Config(circle_vertices=36),
        )

        self.assertTrue(candidates)
        self.assertTrue(
            all(
                evaluate_candidate_domains(
                    q,
                    state,
                    config=Q2Config(circle_vertices=36),
                ).in_c_poss_proxy
                for q in candidates
            )
        )

    def test_detector_candidates_are_not_clipped_to_1800m_arena(self) -> None:
        state = self.make_state(
            (
                (1750.0, -50.0),
                (1850.0, -50.0),
                (1850.0, 50.0),
                (1750.0, 50.0),
            )
        )
        candidates = coarse_grid_candidates(
            state,
            spacing_m=250.0,
            config=Q2Config(circle_vertices=36),
        )

        self.assertTrue(any(math.hypot(*q) > 1800.0 for q in candidates))

    def test_grid_is_deterministic_and_globally_aligned(self) -> None:
        state = self.make_state(
            (
                (110.0, 110.0),
                (210.0, 110.0),
                (210.0, 210.0),
                (110.0, 210.0),
            )
        )
        first = coarse_grid_candidates(
            state,
            spacing_m=200.0,
            config=Q2Config(circle_vertices=36),
        )
        second = coarse_grid_candidates(
            state,
            spacing_m=200.0,
            config=Q2Config(circle_vertices=36),
        )

        self.assertEqual(first, second)
        self.assertTrue(all(q[0] % 200.0 == 0.0 and q[1] % 200.0 == 0.0 for q in first))

    def test_empty_outer_region_fails(self) -> None:
        state = FirstState(
            observation=FirstDirectionObservation((0.0, 0.0), 0.0),
            exact_support_label="empty",
            outer_region=(),
            joint_samples=(JointSample((50.0, 0.0), 1000.0, 1.0),),
        )
        with self.assertRaisesRegex(ValueError, "outer_region"):
            coarse_grid_candidates(state, spacing_m=100.0)


class MinimaxBaselineTests(unittest.TestCase):
    def make_state(self) -> FirstState:
        return FirstState(
            observation=FirstDirectionObservation((0.0, 0.0), 0.0),
            exact_support_label="optimizer test support",
            outer_region=(
                (-100.0, -100.0),
                (100.0, -100.0),
                (100.0, 100.0),
                (-100.0, 100.0),
            ),
            joint_samples=(JointSample((50.0, 0.0), 1000.0, 1.0),),
        )

    @staticmethod
    def result(
        q: tuple[float, float],
        *,
        u_proxy: float,
        u_bar: float,
        movement: float,
        in_c_poss: bool = True,
    ) -> RobustEvaluation:
        return RobustEvaluation(
            q=q,
            u_proxy_m=u_proxy,
            u_bar_m=u_bar,
            worst_response_proxy=None,
            worst_response_outer=None,
            in_c_poss_proxy=in_c_poss,
            in_c_rec_certified=False,
            movement_m=movement,
        )

    def test_selects_smallest_u_proxy_not_smallest_u_bar(self) -> None:
        state = self.make_state()
        candidates = ((1.0, 0.0), (2.0, 0.0))
        results = {
            candidates[0]: self.result(
                candidates[0], u_proxy=10.0, u_bar=100.0, movement=1.0
            ),
            candidates[1]: self.result(
                candidates[1], u_proxy=12.0, u_bar=20.0, movement=2.0
            ),
        }

        with patch(
            "src.q2.optimizer.evaluate_robust",
            side_effect=lambda q, *_args, **_kwargs: results[q],
        ):
            chosen = minimax_baseline(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            )

        self.assertEqual(chosen.q, candidates[0])
        self.assertEqual(chosen.u_proxy_m, 10.0)
        self.assertEqual(chosen.u_bar_m, 100.0)

    def test_movement_breaks_equal_proxy_tie(self) -> None:
        state = self.make_state()
        candidates = ((100.0, 0.0), (10.0, 0.0))
        results = {
            candidates[0]: self.result(
                candidates[0], u_proxy=10.0, u_bar=20.0, movement=100.0
            ),
            candidates[1]: self.result(
                candidates[1], u_proxy=10.0, u_bar=30.0, movement=10.0
            ),
        }

        with patch(
            "src.q2.optimizer.evaluate_robust",
            side_effect=lambda q, *_args, **_kwargs: results[q],
        ):
            chosen = minimax_baseline(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            )

        self.assertEqual(chosen.q, candidates[1])

    def test_coordinates_break_complete_tie_deterministically(self) -> None:
        state = self.make_state()
        candidates = ((1.0, 2.0), (-1.0, 2.0))
        results = {
            q: self.result(q, u_proxy=10.0, u_bar=20.0, movement=5.0)
            for q in candidates
        }

        with patch(
            "src.q2.optimizer.evaluate_robust",
            side_effect=lambda q, *_args, **_kwargs: results[q],
        ):
            chosen = minimax_baseline(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            )

        self.assertEqual(chosen.q, (-1.0, 2.0))

    def test_ignores_candidates_outside_c_poss_proxy(self) -> None:
        state = self.make_state()
        candidates = ((1.0, 0.0), (2.0, 0.0))
        results = {
            candidates[0]: self.result(
                candidates[0],
                u_proxy=1.0,
                u_bar=1.0,
                movement=1.0,
                in_c_poss=False,
            ),
            candidates[1]: self.result(
                candidates[1],
                u_proxy=10.0,
                u_bar=10.0,
                movement=2.0,
                in_c_poss=True,
            ),
        }

        with patch(
            "src.q2.optimizer.evaluate_robust",
            side_effect=lambda q, *_args, **_kwargs: results[q],
        ):
            chosen = minimax_baseline(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            )

        self.assertEqual(chosen.q, candidates[1])

    def test_rejects_empty_or_fully_inadmissible_candidates(self) -> None:
        state = self.make_state()

        with self.assertRaisesRegex(ValueError, "non-empty"):
            minimax_baseline(
                (),
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
            )

        with patch(
            "src.q2.optimizer.evaluate_robust",
            return_value=self.result(
                (1.0, 0.0),
                u_proxy=1.0,
                u_bar=1.0,
                movement=1.0,
                in_c_poss=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "C_poss"):
                minimax_baseline(
                    ((1.0, 0.0),),
                    state,
                    direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                )


class UnifiedScoringTests(unittest.TestCase):
    def make_state(self) -> FirstState:
        return FirstState(
            observation=FirstDirectionObservation((0.0, 0.0), 0.0),
            exact_support_label="optimizer scoring test support",
            outer_region=(
                (-100.0, -100.0),
                (100.0, -100.0),
                (100.0, 100.0),
                (-100.0, 100.0),
            ),
            joint_samples=(JointSample((50.0, 0.0), 1000.0, 1.0),),
        )

    @staticmethod
    def bayes(
        q: tuple[float, float],
        *,
        psi: float,
        movement: float,
    ) -> BayesianEvaluation:
        return BayesianEvaluation(
            q=q,
            psi_d_m=psi,
            response_probabilities={},
            metrics=(),
            movement_m=movement,
        )

    @staticmethod
    def robust(
        q: tuple[float, float],
        *,
        u_proxy: float,
        movement: float,
        in_c_poss: bool = True,
    ) -> RobustEvaluation:
        return RobustEvaluation(
            q=q,
            u_proxy_m=u_proxy,
            u_bar_m=u_proxy + 5.0,
            worst_response_proxy=None,
            worst_response_outer=None,
            in_c_poss_proxy=in_c_poss,
            in_c_rec_certified=False,
            movement_m=movement,
        )

    def test_candidate_score_requires_matching_q(self) -> None:
        with self.assertRaisesRegex(ValueError, "same q"):
            CandidateScore(
                q=(0.0, 0.0),
                bayes=self.bayes((1.0, 0.0), psi=10.0, movement=1.0),
                robust=self.robust((0.0, 0.0), u_proxy=20.0, movement=0.0),
            )

    def test_score_candidates_uses_same_candidates_and_explicit_error_atoms(self) -> None:
        state = self.make_state()
        candidates = ((0.0, 0.0), (50.0, 0.0))
        atoms = (
            BearingErrorAtom(-1.0, 0.25),
            BearingErrorAtom(0.0, 0.50),
            BearingErrorAtom(1.0, 0.25),
        )

        with (
            patch(
                "src.q2.optimizer.evaluate_candidate_domains",
                side_effect=lambda q, *_args, **_kwargs: type(
                    "Flags", (), {"in_c_poss_proxy": True}
                )(),
            ),
            patch(
                "src.q2.optimizer.evaluate_bayesian",
                side_effect=lambda q, *_args, **_kwargs: self.bayes(
                    q, psi=10.0 + q[0], movement=abs(q[0])
                ),
            ) as bayes_eval,
            patch(
                "src.q2.optimizer.evaluate_robust",
                side_effect=lambda q, *_args, **_kwargs: self.robust(
                    q, u_proxy=20.0 + q[0], movement=abs(q[0])
                ),
            ) as robust_eval,
        ):
            scores = score_candidates(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                bearing_error_atoms=atoms,
            )

        self.assertEqual(tuple(score.q for score in scores), candidates)
        self.assertEqual(bayes_eval.call_count, 2)
        self.assertEqual(robust_eval.call_count, 2)
        for call in bayes_eval.call_args_list:
            self.assertEqual(call.kwargs["bearing_error_atoms"], atoms)

    def test_score_candidates_skips_outside_c_poss_before_evaluation(self) -> None:
        state = self.make_state()
        candidates = ((0.0, 0.0), (5000.0, 0.0))
        atoms = (BearingErrorAtom(0.0, 1.0),)

        def flags(q, *_args, **_kwargs):
            return type("Flags", (), {"in_c_poss_proxy": q == candidates[0]})()

        with (
            patch("src.q2.optimizer.evaluate_candidate_domains", side_effect=flags),
            patch(
                "src.q2.optimizer.evaluate_bayesian",
                return_value=self.bayes(candidates[0], psi=10.0, movement=0.0),
            ) as bayes_eval,
            patch(
                "src.q2.optimizer.evaluate_robust",
                return_value=self.robust(candidates[0], u_proxy=20.0, movement=0.0),
            ) as robust_eval,
        ):
            scores = score_candidates(
                candidates,
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                bearing_error_atoms=atoms,
            )

        self.assertEqual(tuple(score.q for score in scores), (candidates[0],))
        self.assertEqual(bayes_eval.call_count, 1)
        self.assertEqual(robust_eval.call_count, 1)

    def test_score_candidates_rejects_empty_and_fully_inadmissible_inputs(self) -> None:
        state = self.make_state()
        atoms = (BearingErrorAtom(0.0, 1.0),)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            score_candidates(
                (),
                state,
                direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                bearing_error_atoms=atoms,
            )

        with patch(
            "src.q2.optimizer.evaluate_candidate_domains",
            return_value=type("Flags", (), {"in_c_poss_proxy": False})(),
        ):
            with self.assertRaisesRegex(ValueError, "C_poss"):
                score_candidates(
                    ((5000.0, 0.0),),
                    state,
                    direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                    bearing_error_atoms=atoms,
                )

    def test_score_candidates_detects_domain_inconsistency(self) -> None:
        state = self.make_state()
        q = (0.0, 0.0)
        atoms = (BearingErrorAtom(0.0, 1.0),)

        with (
            patch(
                "src.q2.optimizer.evaluate_candidate_domains",
                return_value=type("Flags", (), {"in_c_poss_proxy": True})(),
            ),
            patch(
                "src.q2.optimizer.evaluate_bayesian",
                return_value=self.bayes(q, psi=10.0, movement=0.0),
            ),
            patch(
                "src.q2.optimizer.evaluate_robust",
                return_value=self.robust(
                    q, u_proxy=20.0, movement=0.0, in_c_poss=False
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "inconsistency"):
                score_candidates(
                    (q,),
                    state,
                    direction_grid_deg=(0.0, 90.0, 180.0, 270.0),
                    bearing_error_atoms=atoms,
                )


class PureBayesianSelectorTests(unittest.TestCase):
    @staticmethod
    def score(
        q: tuple[float, float],
        *,
        psi: float,
        movement: float,
    ) -> CandidateScore:
        bayes = BayesianEvaluation(
            q=q,
            psi_d_m=psi,
            response_probabilities={},
            metrics=(),
            movement_m=movement,
        )
        robust = RobustEvaluation(
            q=q,
            u_proxy_m=100.0,
            u_bar_m=110.0,
            worst_response_proxy=None,
            worst_response_outer=None,
            in_c_poss_proxy=True,
            in_c_rec_certified=False,
            movement_m=movement,
        )
        return CandidateScore(q=q, bayes=bayes, robust=robust)

    def test_selects_smallest_psi_when_tau_zero(self) -> None:
        scores = (
            self.score((10.0, 0.0), psi=10.0, movement=10.0),
            self.score((1.0, 0.0), psi=11.0, movement=1.0),
        )
        chosen = select_pure_bayesian(scores)
        self.assertEqual(chosen.q, (10.0, 0.0))

    def test_tau_allows_shorter_movement_among_near_optimal_scores(self) -> None:
        scores = (
            self.score((10.0, 0.0), psi=10.0, movement=10.0),
            self.score((1.0, 0.0), psi=10.4, movement=1.0),
            self.score((0.5, 0.0), psi=10.8, movement=0.5),
        )
        chosen = select_pure_bayesian(scores, tau_m=0.5)
        self.assertEqual(chosen.q, (1.0, 0.0))

    def test_coordinates_break_complete_tie_deterministically(self) -> None:
        scores = (
            self.score((1.0, 2.0), psi=10.0, movement=5.0),
            self.score((-1.0, 2.0), psi=10.0, movement=5.0),
        )
        chosen = select_pure_bayesian(scores)
        self.assertEqual(chosen.q, (-1.0, 2.0))

    def test_rejects_invalid_tau_and_empty_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            select_pure_bayesian(())

        score = self.score((0.0, 0.0), psi=10.0, movement=0.0)
        for tau in (-1.0, math.inf, math.nan, True):
            with self.subTest(tau=tau):
                with self.assertRaises(ValueError):
                    select_pure_bayesian((score,), tau_m=tau)  # type: ignore[arg-type]



class RobustEnvelopeHybridTests(unittest.TestCase):
    @staticmethod
    def score(
        q: tuple[float, float],
        *,
        psi: float,
        u_proxy: float,
        u_bar: float | None = None,
        movement: float,
    ) -> CandidateScore:
        bayes = BayesianEvaluation(
            q=q,
            psi_d_m=psi,
            response_probabilities={},
            metrics=(),
            movement_m=movement,
        )
        robust = RobustEvaluation(
            q=q,
            u_proxy_m=u_proxy,
            u_bar_m=u_proxy if u_bar is None else u_bar,
            worst_response_proxy=None,
            worst_response_outer=None,
            in_c_poss_proxy=True,
            in_c_rec_certified=False,
            movement_m=movement,
        )
        return CandidateScore(q=q, bayes=bayes, robust=robust)

    def test_unified_pure_minimax_matches_proxy_then_movement_rule(self) -> None:
        scores = (
            self.score((100.0, 0.0), psi=1.0, u_proxy=10.0, movement=100.0),
            self.score((10.0, 0.0), psi=9.0, u_proxy=10.0, movement=10.0),
            self.score((1.0, 0.0), psi=0.1, u_proxy=11.0, movement=1.0),
        )

        chosen = select_pure_minimax(scores)

        self.assertEqual(chosen.q, (10.0, 0.0))

    def test_rho_zero_reduces_hybrid_to_minimax_envelope(self) -> None:
        scores = (
            self.score((10.0, 0.0), psi=20.0, u_proxy=10.0, movement=10.0),
            self.score((20.0, 0.0), psi=1.0, u_proxy=11.0, movement=20.0),
        )

        chosen = select_robust_envelope_hybrid(scores, rho=0.0)

        self.assertEqual(chosen.q, (10.0, 0.0))

    def test_positive_rho_admits_better_bayesian_point(self) -> None:
        scores = (
            self.score((10.0, 0.0), psi=20.0, u_proxy=10.0, movement=10.0),
            self.score((20.0, 0.0), psi=1.0, u_proxy=11.0, movement=20.0),
            self.score((30.0, 0.0), psi=0.1, u_proxy=13.0, movement=30.0),
        )

        chosen = select_robust_envelope_hybrid(scores, rho=0.10)

        self.assertEqual(chosen.q, (20.0, 0.0))

    def test_tau_uses_movement_only_inside_robust_envelope(self) -> None:
        scores = (
            self.score((10.0, 0.0), psi=10.0, u_proxy=10.0, movement=10.0),
            self.score((1.0, 0.0), psi=10.3, u_proxy=10.5, movement=1.0),
            self.score((0.5, 0.0), psi=10.4, u_proxy=13.0, movement=0.5),
        )

        chosen = select_robust_envelope_hybrid(
            scores,
            rho=0.05,
            tau_m=0.5,
        )

        self.assertEqual(chosen.q, (1.0, 0.0))

    def test_outer_envelope_can_select_differently_from_proxy_envelope(self) -> None:
        scores = (
            self.score(
                (1.0, 0.0),
                psi=5.0,
                u_proxy=10.0,
                u_bar=30.0,
                movement=1.0,
            ),
            self.score(
                (2.0, 0.0),
                psi=1.0,
                u_proxy=11.0,
                u_bar=20.0,
                movement=2.0,
            ),
        )

        proxy_choice = select_robust_envelope_hybrid(scores, rho=0.0)
        outer_choice = select_robust_envelope_hybrid(
            scores,
            rho=0.0,
            use_outer_envelope=True,
        )

        self.assertEqual(proxy_choice.q, (1.0, 0.0))
        self.assertEqual(outer_choice.q, (2.0, 0.0))

    def test_boundary_point_at_one_plus_rho_is_included(self) -> None:
        scores = (
            self.score((1.0, 0.0), psi=10.0, u_proxy=100.0, movement=1.0),
            self.score((2.0, 0.0), psi=1.0, u_proxy=110.0, movement=2.0),
        )

        chosen = select_robust_envelope_hybrid(scores, rho=0.10)

        self.assertEqual(chosen.q, (2.0, 0.0))

    def test_rejects_invalid_rho_tau_and_empty_scores(self) -> None:
        score = self.score((0.0, 0.0), psi=1.0, u_proxy=1.0, movement=0.0)

        with self.assertRaisesRegex(ValueError, "non-empty"):
            select_robust_envelope_hybrid((), rho=0.1)

        for rho in (-0.1, math.inf, math.nan, True):
            with self.subTest(rho=rho):
                with self.assertRaises(ValueError):
                    select_robust_envelope_hybrid(
                        (score,),
                        rho=rho,  # type: ignore[arg-type]
                    )

        for tau in (-0.1, math.inf, math.nan, True):
            with self.subTest(tau=tau):
                with self.assertRaises(ValueError):
                    select_robust_envelope_hybrid(
                        (score,),
                        rho=0.1,
                        tau_m=tau,  # type: ignore[arg-type]
                    )

    def test_rejects_nonfinite_or_negative_metrics(self) -> None:
        bad_robust = self.score(
            (0.0, 0.0),
            psi=1.0,
            u_proxy=math.inf,
            movement=0.0,
        )
        with self.assertRaisesRegex(ValueError, "robust envelope metric"):
            select_robust_envelope_hybrid((bad_robust,), rho=0.1)

        bad_bayes = self.score(
            (0.0, 0.0),
            psi=math.inf,
            u_proxy=1.0,
            movement=0.0,
        )
        with self.assertRaisesRegex(ValueError, "Bayesian score"):
            select_robust_envelope_hybrid((bad_bayes,), rho=0.1)



if __name__ == "__main__":
    unittest.main()
