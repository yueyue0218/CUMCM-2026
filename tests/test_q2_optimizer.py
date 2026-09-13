import math
import unittest
from unittest.mock import patch

from src.q2.model import (
    FirstDirectionObservation,
    FirstState,
    JointSample,
    Q2Config,
    RobustEvaluation,
)
from src.q2.optimizer import coarse_grid_candidates, minimax_baseline


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


if __name__ == "__main__":
    unittest.main()
