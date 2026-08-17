from __future__ import annotations

import random
import unittest
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from lolo_agent.accessibility import (
    BRANCH_EVENT,
    CERTIFICATION_CENSORED,
    CERTIFIED_HELD,
    CONFIGURATION_DEPARTED,
    CertificationWindow,
    CertifiedCoverage,
    GRID_COLUMNS,
    ProbeBudget,
    REASON_CONFIRMED_EFFECT_SIGNATURE_DEPARTED,
    REASON_MISSING_TRACK_KEYS,
    REASON_POST_CAUSAL_RESTORE,
    REASON_TRACKED_STATE_SIGNATURE_DEPARTED,
    REASON_TRACK_CELLS_DEPARTED,
    REASON_UNORDERED_AGAINST_RESTORE,
    RootTrackState,
    band_cells,
    branch_endpoint_cell,
    certification_window,
    certify_branch,
    coverage_from_branches,
    delta,
    jaccard,
    repetition_agreement,
    score_target_bit,
)

Cell = Tuple[int, int]

# ---------------------------------------------------------------------------
# Fixtures: small real record shapes copied from the object-removed probe
# analyses (docs/object-removed-probe-2026-08-16.md, run entity-v325).  The
# dicts restate what the probe telemetry actually contains, trimmed to the
# keys this instrument reads; tests never read the gitignored experiments/
# archives themselves.
# ---------------------------------------------------------------------------

V325_ROOT_STATE_SHA = (
    "bdb5bbde46acbd44dde775dec02d24a2ac9b1efe329967c0251e436c9e6b0d49"
)
V324_PREPUSH_CHECKPOINT_SHA = (
    "33addc6c00000000000000000000000000000000000000000000000000000fb92"
)

# v325 decision-1 commit: the resume decision event carries no track keys,
# so the root track seeds empty (legacy_track_reconstructed=true).
V325_ROOT_RECORD: Dict[str, Any] = {
    "event": "decision_committed",
    "seq": 14636,
    "decision": 1,
    "anonymous_object_track_cells": [],
    "anonymous_object_track_current_cell": None,
    "anonymous_object_track_confirmed_source_cell": None,
    "anonymous_object_track_confirmed_destination_cell": None,
    "anonymous_object_track_confirmed_world_effect_signature": None,
    "human_prior_option_tracked_world_state_signature": None,
}

# v325 seq 77: a certified configuration-held branch (cells == [], both
# signatures null), endpoint slot (112, 96) -> coarse cell (7, 6).
V325_CLEAN_BRANCH: Dict[str, Any] = {
    "event": "human_prior_option_branch_verified",
    "seq": 77,
    "decision": 1,
    "branch_index": 1,
    "depth": 1,
    "path": ["up"],
    "durations": [16],
    "parent_state_id": "state-00000003",
    "state_id": "state-00000004",
    "source_state_id": "state-00000003",
    "frame": "f63a9a94cc01922de962acc265872c67335b6d35d9e054dafd6e1b0c8a4341eb",
    "frame_width": 256,
    "frame_height": 240,
    "human_prior_target_player_slot": [112, 96],
    "human_prior_option_tracked_world_effect_cells": [],
    "human_prior_option_tracked_world_state_signature": None,
    "anonymous_object_track_cells": [],
    "anonymous_object_track_current_cell": None,
    "anonymous_object_track_confirmed_source_cell": None,
    "anonymous_object_track_confirmed_destination_cell": None,
    "anonymous_object_track_confirmed_world_effect_signature": None,
    "human_prior_option_nonlocal_world_effect_cells": [],
}

# v325 seq 6286: a configuration-departed branch (accumulated track cell
# (12, 5), tracked world-state signature set).
V325_DEPARTED_BRANCH: Dict[str, Any] = {
    "event": "human_prior_option_branch_verified",
    "seq": 6286,
    "decision": 1,
    "branch_index": 1351,
    "depth": 10,
    "path": [
        "right", "down", "down", "right", "right",
        "up", "up", "right", "right", "a",
    ],
    "durations": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    "parent_state_id": "state-00001330",
    "state_id": "state-00001354",
    "source_state_id": "state-00000003",
    "frame": "510983a6994cd4aae137fd01cc0aa50098d9f9d2e4f930f94cc349d7dc60549a",
    "frame_width": 256,
    "frame_height": 240,
    "human_prior_target_player_slot": [192, 96],
    "human_prior_option_tracked_world_effect_cells": [[12, 5]],
    "human_prior_option_tracked_world_state_signature": "2043852bb0ab4bb3",
    "anonymous_object_track_cells": [[12, 5]],
    "anonymous_object_track_current_cell": [12, 5],
    "anonymous_object_track_confirmed_source_cell": None,
    "anonymous_object_track_confirmed_destination_cell": None,
    "anonymous_object_track_confirmed_world_effect_signature": None,
    "human_prior_option_nonlocal_world_effect_cells": [],
}

# v325's first causal-archive restore (seq 15054, decision 2): carries no
# track fields, which is exactly why it voids later certification.
V325_CAUSAL_RESTORE: Dict[str, Any] = {
    "event": "archive_branch_restored",
    "seq": 15054,
    "decision": 2,
    "action": "right",
    "action_frames": 16,
    "archive_size": 80,
}

# Certified coverage envelopes from the probe results (documented cell
# sets): both certified baselines (pushed v322, pre-push v324) share the
# identical 7-cell envelope; the object-removed configuration (v325, and
# its repetition v326 at Jaccard 1.0) certified 24 cells.
BASELINE_ENVELOPE: Tuple[Cell, ...] = (
    (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 10), (8, 10),
)
REMOVED_ENVELOPE: Tuple[Cell, ...] = (
    (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 6), (7, 10), (8, 6),
    (8, 7), (8, 8), (8, 10), (9, 8), (10, 6), (10, 7), (10, 8), (11, 6),
    (11, 7), (11, 8), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10),
    (12, 11),
)
# The removed entity's home and the transformed object's transit cell: the
# declared footprint the probes exclude from every delta claim.
FOOTPRINT: Tuple[Cell, ...] = ((7, 6), (8, 6))
COLUMN8_BAND: Tuple[Cell, ...] = ((8, 7), (8, 8), (9, 8))


def _slot(cell: Cell) -> Tuple[int, int]:
    # Inverse of the coarse-cell derivation for 256x240 frames: 16 pixels
    # per column, 16 per row.
    return (cell[0] * 16, cell[1] * 16)


def _branch(
    cell: Optional[Cell],
    *,
    seq: int = 77,
    decision: int = 1,
    cells: Sequence[Cell] = (),
    tracked_signature: Optional[str] = None,
    confirmed_signature: Optional[str] = None,
) -> Dict[str, Any]:
    """A branch record with the real event shape at a chosen endpoint."""

    record = dict(V325_CLEAN_BRANCH)
    record["seq"] = seq
    record["decision"] = decision
    record["anonymous_object_track_cells"] = [list(c) for c in cells]
    record["human_prior_option_tracked_world_state_signature"] = (
        tracked_signature
    )
    record["anonymous_object_track_confirmed_world_effect_signature"] = (
        confirmed_signature
    )
    if cell is None:
        record.pop("human_prior_target_player_slot", None)
    else:
        record["human_prior_target_player_slot"] = list(_slot(cell))
    return record


def _coverage(
    cells: Iterable[Cell],
    *,
    root_state_signature: str,
    seq_start: int = 100,
) -> CertifiedCoverage:
    records = [
        _branch(cell, seq=seq_start + index)
        for index, cell in enumerate(sorted(cells))
    ]
    return coverage_from_branches(
        records,
        V325_ROOT_RECORD,
        root_state_signature=root_state_signature,
    )


class CertifyBranchTests(unittest.TestCase):
    def test_real_clean_branch_certifies_against_empty_root(self) -> None:
        result = certify_branch(V325_CLEAN_BRANCH, V325_ROOT_RECORD)
        self.assertEqual(result.status, CERTIFIED_HELD)
        self.assertTrue(result.certified)
        self.assertEqual(result.reasons, ())

    def test_real_departed_branch_reports_every_departure(self) -> None:
        result = certify_branch(V325_DEPARTED_BRANCH, V325_ROOT_RECORD)
        self.assertEqual(result.status, CONFIGURATION_DEPARTED)
        self.assertIn(REASON_TRACK_CELLS_DEPARTED, result.reasons)
        self.assertIn(
            REASON_TRACKED_STATE_SIGNATURE_DEPARTED, result.reasons
        )
        self.assertNotIn(
            REASON_CONFIRMED_EFFECT_SIGNATURE_DEPARTED, result.reasons
        )

    def test_predicate_truth_table(self) -> None:
        # Certified iff cells AND tracked signature AND confirmed
        # signature all match the root; any single departure suffices.
        cases = [
            (cells_match, tracked_match, confirmed_match)
            for cells_match in (True, False)
            for tracked_match in (True, False)
            for confirmed_match in (True, False)
        ]
        for cells_match, tracked_match, confirmed_match in cases:
            record = _branch(
                (7, 6),
                cells=() if cells_match else ((12, 5),),
                tracked_signature=None if tracked_match else "2043852b",
                confirmed_signature=None if confirmed_match else "deadbeef",
            )
            result = certify_branch(record, V325_ROOT_RECORD)
            expected_held = cells_match and tracked_match and confirmed_match
            with self.subTest(
                cells=cells_match,
                tracked=tracked_match,
                confirmed=confirmed_match,
            ):
                self.assertEqual(
                    result.status,
                    CERTIFIED_HELD
                    if expected_held
                    else CONFIGURATION_DEPARTED,
                )
                self.assertEqual(
                    REASON_TRACK_CELLS_DEPARTED in result.reasons,
                    not cells_match,
                )
                self.assertEqual(
                    REASON_TRACKED_STATE_SIGNATURE_DEPARTED
                    in result.reasons,
                    not tracked_match,
                )
                self.assertEqual(
                    REASON_CONFIRMED_EFFECT_SIGNATURE_DEPARTED
                    in result.reasons,
                    not confirmed_match,
                )

    def test_nonempty_root_certifies_hold_and_flags_disappearance(
        self,
    ) -> None:
        root = {
            "anonymous_object_track_cells": [[7, 6]],
            "human_prior_option_tracked_world_state_signature": "roothash",
            "anonymous_object_track_confirmed_world_effect_signature": None,
        }
        held = _branch(
            (6, 9), cells=((7, 6),), tracked_signature="roothash"
        )
        self.assertEqual(
            certify_branch(held, root).status, CERTIFIED_HELD
        )
        # Track disappearance is a departure, not a hold: the disclosed
        # respawn/return risk is caught as departure by the same predicate.
        vanished = _branch((6, 9), cells=(), tracked_signature="roothash")
        self.assertEqual(
            certify_branch(vanished, root).status, CONFIGURATION_DEPARTED
        )

    def test_cell_order_is_insensitive(self) -> None:
        root = {
            "anonymous_object_track_cells": [[8, 6], [7, 6]],
            "human_prior_option_tracked_world_state_signature": "roothash",
        }
        record = _branch(
            (6, 9),
            cells=((7, 6), (8, 6)),
            tracked_signature="roothash",
        )
        self.assertEqual(
            certify_branch(record, root).status, CERTIFIED_HELD
        )

    def test_null_empty_and_missing_signatures_are_equivalent(self) -> None:
        record = _branch((7, 6))
        record["human_prior_option_tracked_world_state_signature"] = ""
        del record["anonymous_object_track_confirmed_world_effect_signature"]
        self.assertEqual(
            certify_branch(record, V325_ROOT_RECORD).status, CERTIFIED_HELD
        )

    def test_branch_missing_track_keys_is_censored_not_departed(self) -> None:
        # Pre-instrument-fix telemetry (v322/v323 era) cannot certify: the
        # branch is censored, never classified in either direction.
        record = dict(V325_CLEAN_BRANCH)
        del record["anonymous_object_track_cells"]
        result = certify_branch(record, V325_ROOT_RECORD)
        self.assertEqual(result.status, CERTIFICATION_CENSORED)
        self.assertEqual(result.reasons, (REASON_MISSING_TRACK_KEYS,))


class CertificationWindowTests(unittest.TestCase):
    def test_window_locates_first_causal_restore(self) -> None:
        later_restore = dict(V325_CAUSAL_RESTORE, seq=20000, decision=4)
        window = certification_window(
            [V325_CLEAN_BRANCH, V325_CAUSAL_RESTORE, later_restore]
        )
        self.assertTrue(window.bounded)
        self.assertEqual(window.restore_seq, 15054)
        self.assertEqual(window.restore_decision, 2)

    def test_window_unbounded_without_restore(self) -> None:
        window = certification_window(
            [V325_CLEAN_BRANCH, V325_DEPARTED_BRANCH]
        )
        self.assertFalse(window.bounded)
        self.assertEqual(
            certify_branch(
                V325_CLEAN_BRANCH, V325_ROOT_RECORD, window=window
            ).status,
            CERTIFIED_HELD,
        )

    def test_pre_restore_branch_certifies_post_restore_is_censored(
        self,
    ) -> None:
        window = certification_window([V325_CAUSAL_RESTORE])
        pre = _branch((7, 6), seq=15053)
        post = _branch((7, 6), seq=15055, decision=2)
        self.assertEqual(
            certify_branch(pre, V325_ROOT_RECORD, window=window).status,
            CERTIFIED_HELD,
        )
        result = certify_branch(post, V325_ROOT_RECORD, window=window)
        self.assertEqual(result.status, CERTIFICATION_CENSORED)
        self.assertEqual(result.reasons, (REASON_POST_CAUSAL_RESTORE,))

    def test_post_restore_departure_is_also_censored(self) -> None:
        # After the tracker was silently reset, apparent departures are as
        # untrustworthy as apparent holds: reported, never classified.
        window = certification_window([V325_CAUSAL_RESTORE])
        post = _branch(
            (8, 7),
            seq=16000,
            decision=3,
            cells=((12, 5),),
            tracked_signature="2043852bb0ab4bb3",
        )
        result = certify_branch(post, V325_ROOT_RECORD, window=window)
        self.assertEqual(result.status, CERTIFICATION_CENSORED)

    def test_decision_ordering_fallback_without_seq(self) -> None:
        window = certification_window([V325_CAUSAL_RESTORE])
        pre = _branch((7, 6), decision=1)
        del pre["seq"]
        post = _branch((7, 6), decision=2)
        del post["seq"]
        self.assertEqual(
            certify_branch(pre, V325_ROOT_RECORD, window=window).status,
            CERTIFIED_HELD,
        )
        self.assertEqual(
            certify_branch(post, V325_ROOT_RECORD, window=window).status,
            CERTIFICATION_CENSORED,
        )

    def test_unorderable_branch_is_conservatively_censored(self) -> None:
        window = CertificationWindow(restore_seq=15054)
        record = _branch((7, 6))
        del record["seq"]
        del record["decision"]
        result = certify_branch(record, V325_ROOT_RECORD, window=window)
        self.assertEqual(result.status, CERTIFICATION_CENSORED)
        self.assertEqual(
            result.reasons, (REASON_UNORDERED_AGAINST_RESTORE,)
        )


class CoverageTests(unittest.TestCase):
    def test_tiers_counts_and_branch_counts(self) -> None:
        records = [
            V325_CAUSAL_RESTORE,
            _branch((6, 6), seq=100),
            _branch((6, 6), seq=101),
            _branch((7, 10), seq=102),
            # Departed branch reaching a certified cell: the cell stays in
            # the certified tier only.
            _branch(
                (6, 6),
                seq=103,
                cells=((12, 5),),
                tracked_signature="2043852bb0ab4bb3",
            ),
            # Departed-only cell.
            _branch(
                (8, 7),
                seq=104,
                cells=((12, 5),),
                tracked_signature="2043852bb0ab4bb3",
            ),
            # Censored-only cell (post-restore, clean-looking fields).
            _branch((9, 8), seq=15100, decision=2),
            # Certified branch without a detected endpoint slot.
            _branch(None, seq=105),
        ]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
            budget=ProbeBudget(search_depth=12, beam_width=128),
        )
        self.assertEqual(coverage.certified_cells, ((6, 6), (7, 10)))
        self.assertEqual(
            coverage.certified_cell_branch_counts,
            (((6, 6), 2), ((7, 10), 1)),
        )
        self.assertEqual(coverage.side_effect_only_cells, ((8, 7),))
        self.assertEqual(coverage.certification_censored_cells, ((9, 8),))
        self.assertEqual(coverage.branches_total, 7)
        self.assertEqual(coverage.branches_certified, 4)
        self.assertEqual(coverage.branches_departed, 2)
        self.assertEqual(coverage.branches_censored, 1)
        self.assertEqual(coverage.branches_without_endpoint_cell, 1)
        self.assertEqual(coverage.window.restore_seq, 15054)
        self.assertEqual(coverage.budget.search_depth, 12)

    def test_censored_reach_is_distinct_from_absent(self) -> None:
        records = [
            V325_CAUSAL_RESTORE,
            _branch((6, 6), seq=100),
            _branch((9, 8), seq=15100, decision=2),
        ]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        # (9, 8) was reached but its certification is censored; (5, 5) was
        # never reached at all.  The two must remain distinguishable.
        self.assertIn((9, 8), coverage.certification_censored_cells)
        self.assertNotIn((9, 8), coverage.certified_cells)
        for tier in (
            coverage.certified_cells,
            coverage.side_effect_only_cells,
            coverage.certification_censored_cells,
        ):
            self.assertNotIn((5, 5), tier)

    def test_explicit_window_overrides_stream_derivation(self) -> None:
        # Analyzing a filtered stream with the full run's window: the
        # branch stream alone shows no restore, but the explicit window
        # still censors post-restore branches.
        records = [_branch((9, 8), seq=15100, decision=2)]
        window = certification_window([V325_CAUSAL_RESTORE])
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
            window=window,
        )
        self.assertEqual(coverage.certified_cells, ())
        self.assertEqual(coverage.certification_censored_cells, ((9, 8),))

    def test_non_branch_events_are_ignored_for_coverage(self) -> None:
        records = [
            V325_ROOT_RECORD,
            V325_CAUSAL_RESTORE,
            _branch((6, 6), seq=100),
        ]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        self.assertEqual(coverage.branches_total, 1)
        self.assertEqual(coverage.certified_cells, ((6, 6),))


class DeltaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.before = _coverage(
            BASELINE_ENVELOPE,
            root_state_signature=V324_PREPUSH_CHECKPOINT_SHA,
        )
        self.after = _coverage(
            REMOVED_ENVELOPE,
            root_state_signature=V325_ROOT_STATE_SHA,
        )

    def test_gate3_shape_delta_on_the_real_envelopes(self) -> None:
        measured = delta(
            self.before,
            self.after,
            excluded_footprint_cells=FOOTPRINT,
        )
        expected_new = tuple(
            sorted(
                set(REMOVED_ENVELOPE)
                - set(BASELINE_ENVELOPE)
                - set(FOOTPRINT)
            )
        )
        self.assertEqual(measured.newly_reachable_cells, expected_new)
        self.assertEqual(len(measured.newly_reachable_cells), 15)
        self.assertIn((12, 11), measured.newly_reachable_cells)
        for band_cell in COLUMN8_BAND:
            self.assertIn(band_cell, measured.newly_reachable_cells)
        self.assertEqual(
            measured.shared_certified_cells, BASELINE_ENVELOPE
        )
        self.assertEqual(measured.no_longer_reached_cells, ())
        self.assertEqual(
            measured.source_state_signature, V324_PREPUSH_CHECKPOINT_SHA
        )
        self.assertEqual(
            measured.target_state_signature, V325_ROOT_STATE_SHA
        )
        self.assertTrue(measured.non_reach_censored)

    def test_trivial_vacated_cell_trap_is_excluded_by_declaration(
        self,
    ) -> None:
        # Without the declared footprint the vacated home (7, 6) and the
        # transit cell (8, 6) would inflate the delta -- the trivially
        # nonzero vacated-cell claim the preregistrations forbid.
        undeclared = delta(
            self.before, self.after, excluded_footprint_cells=()
        )
        self.assertIn((7, 6), undeclared.newly_reachable_cells)
        self.assertIn((8, 6), undeclared.newly_reachable_cells)
        declared = delta(
            self.before, self.after, excluded_footprint_cells=FOOTPRINT
        )
        for cell in FOOTPRINT:
            self.assertNotIn(cell, declared.newly_reachable_cells)
            self.assertNotIn(cell, declared.shared_certified_cells)
            self.assertNotIn(cell, declared.no_longer_reached_cells)
        self.assertEqual(declared.excluded_footprint_cells, FOOTPRINT)

    def test_footprint_must_be_declared_explicitly(self) -> None:
        with self.assertRaises(TypeError):
            delta(self.before, self.after)  # type: ignore[call-arg]

    def test_footprint_excluded_from_every_cell_field(self) -> None:
        footprint = ((1, 1), (2, 2), (3, 3), (4, 4))
        before = CertifiedCoverage(
            root_state_signature="before",
            root_track_state=RootTrackState((), "", ""),
            window=CertificationWindow(),
            certified_cells=((1, 1), (5, 5)),
            certified_cell_branch_counts=(((1, 1), 1), ((5, 5), 1)),
            side_effect_only_cells=((2, 2), (6, 6)),
            certification_censored_cells=((3, 3), (7, 7)),
            branches_total=6,
            branches_certified=2,
            branches_departed=2,
            branches_censored=2,
            branches_without_endpoint_cell=0,
        )
        after = CertifiedCoverage(
            root_state_signature="after",
            root_track_state=RootTrackState((), "", ""),
            window=CertificationWindow(),
            certified_cells=((4, 4), (8, 8)),
            certified_cell_branch_counts=(((4, 4), 1), ((8, 8), 1)),
            side_effect_only_cells=((2, 2), (9, 9)),
            certification_censored_cells=((3, 3), (10, 10)),
            branches_total=6,
            branches_certified=2,
            branches_departed=2,
            branches_censored=2,
            branches_without_endpoint_cell=0,
        )
        measured = delta(
            before, after, excluded_footprint_cells=footprint
        )
        every_cell_field = (
            measured.newly_reachable_cells
            + measured.no_longer_reached_cells
            + measured.shared_certified_cells
            + measured.source_side_effect_only_cells
            + measured.target_side_effect_only_cells
            + measured.source_certification_censored_cells
            + measured.target_certification_censored_cells
        )
        for cell in footprint:
            self.assertNotIn(cell, every_cell_field)
        self.assertEqual(measured.newly_reachable_cells, ((8, 8),))
        self.assertEqual(measured.no_longer_reached_cells, ((5, 5),))
        self.assertEqual(
            measured.target_side_effect_only_cells, ((9, 9),)
        )
        self.assertEqual(
            measured.target_certification_censored_cells, ((10, 10),)
        )

    def test_budgets_and_provenance_are_carried(self) -> None:
        before = _coverage(
            BASELINE_ENVELOPE,
            root_state_signature=V324_PREPUSH_CHECKPOINT_SHA,
        )
        after_records = [
            _branch(cell, seq=200 + index)
            for index, cell in enumerate(REMOVED_ENVELOPE)
        ]
        after = coverage_from_branches(
            after_records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
            budget=ProbeBudget(
                search_depth=12,
                beam_width=128,
                decisions=8,
                wall_clock_seconds=1462.0,
                wall_clock_ceiling_seconds=10800.0,
                event_count=200000,
                completed_within_ceilings=True,
            ),
        )
        measured = delta(
            before, after, excluded_footprint_cells=FOOTPRINT
        )
        self.assertEqual(measured.target_budget.search_depth, 12)
        self.assertEqual(
            measured.target_budget.wall_clock_ceiling_seconds, 10800.0
        )
        self.assertEqual(
            measured.source_coverage_signature, before.signature
        )
        self.assertEqual(
            measured.target_coverage_signature, after.signature
        )


class ScoredBitTests(unittest.TestCase):
    def test_band_cells_enumerates_the_declared_band(self) -> None:
        band = band_cells(range(8, 10), range(7, 9))
        self.assertEqual(
            band, ((8, 7), (8, 8), (9, 7), (9, 8))
        )
        wide = band_cells(range(8, GRID_COLUMNS), range(5, 8))
        self.assertIn((8, 5), wide)
        self.assertIn((15, 7), wide)
        self.assertNotIn((7, 6), wide)

    def test_band_bit_scores_yes_with_exact_branch_count(self) -> None:
        records = [
            _branch((8, 7), seq=248 + index) for index in range(100)
        ]
        records += [_branch((8, 8), seq=400 + index) for index in range(25)]
        records += [_branch((9, 8), seq=500 + index) for index in range(10)]
        records += [_branch((6, 6), seq=600)]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        bit = score_target_bit(coverage, COLUMN8_BAND)
        self.assertTrue(bit.reached)
        self.assertFalse(bit.non_reach_censored)
        self.assertEqual(bit.certified_branch_count, 135)
        self.assertEqual(bit.certified_cells_reached, COLUMN8_BAND)
        self.assertEqual(bit.coverage_signature, coverage.signature)

    def test_non_reach_is_censored_never_unreachable(self) -> None:
        coverage = _coverage(
            BASELINE_ENVELOPE,
            root_state_signature=V324_PREPUSH_CHECKPOINT_SHA,
        )
        bit = score_target_bit(coverage, COLUMN8_BAND)
        self.assertFalse(bit.reached)
        self.assertTrue(bit.non_reach_censored)
        self.assertEqual(bit.certified_branch_count, 0)
        self.assertEqual(bit.certified_cells_reached, ())

    def test_departed_reach_never_scores_the_bit(self) -> None:
        records = [
            _branch(
                (8, 7),
                seq=100,
                cells=((12, 5),),
                tracked_signature="2043852bb0ab4bb3",
            )
        ]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        bit = score_target_bit(coverage, COLUMN8_BAND)
        self.assertFalse(bit.reached)

    def test_footprint_overlap_cannot_trivialize_the_bit(self) -> None:
        records = [_branch((7, 6), seq=100)]
        coverage = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        bit = score_target_bit(
            coverage,
            ((7, 6), (8, 7)),
            excluded_footprint_cells=FOOTPRINT,
        )
        self.assertEqual(bit.scored_cells, ((8, 7),))
        self.assertFalse(bit.reached)


class RepetitionTests(unittest.TestCase):
    def test_jaccard_values(self) -> None:
        self.assertEqual(jaccard((), ()), 1.0)
        self.assertEqual(jaccard(((1, 1),), ((2, 2),)), 0.0)
        self.assertAlmostEqual(
            jaccard(
                ((1, 1), (2, 2), (3, 3)), ((2, 2), (3, 3), (4, 4))
            ),
            0.5,
        )

    def test_identical_envelopes_close_the_repetition_gate(self) -> None:
        first = _coverage(
            REMOVED_ENVELOPE, root_state_signature=V325_ROOT_STATE_SHA
        )
        second = _coverage(
            REMOVED_ENVELOPE,
            root_state_signature=V325_ROOT_STATE_SHA,
            seq_start=500,
        )
        agreement = repetition_agreement(first, second)
        self.assertEqual(agreement.jaccard, 1.0)
        self.assertTrue(agreement.agreed)
        self.assertEqual(agreement.threshold, 0.8)
        self.assertEqual(agreement.shared_cells, REMOVED_ENVELOPE)
        self.assertEqual(agreement.only_first_cells, ())
        self.assertEqual(agreement.only_second_cells, ())

    def test_divergence_is_reported_and_scoped_not_hidden(self) -> None:
        first = _coverage(
            ((1, 1), (2, 2), (3, 3), (4, 4)),
            root_state_signature="root",
        )
        second = _coverage(
            ((1, 1), (2, 2), (5, 5)),
            root_state_signature="root",
        )
        agreement = repetition_agreement(first, second)
        self.assertAlmostEqual(agreement.jaccard, 2 / 5)
        self.assertFalse(agreement.agreed)
        self.assertEqual(agreement.only_first_cells, ((3, 3), (4, 4)))
        self.assertEqual(agreement.only_second_cells, ((5, 5),))

    def test_threshold_boundary_is_inclusive(self) -> None:
        # Jaccard exactly at the preregistered threshold closes the gate
        # ("substantial agreement (>= 80% Jaccard)").
        first = _coverage(
            ((1, 1), (2, 2), (3, 3), (4, 4)),
            root_state_signature="root",
        )
        second = _coverage(
            ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5)),
            root_state_signature="root",
        )
        agreement = repetition_agreement(first, second)
        self.assertAlmostEqual(agreement.jaccard, 0.8)
        self.assertTrue(agreement.agreed)


class SignatureTests(unittest.TestCase):
    def test_coverage_signature_is_order_invariant_and_deterministic(
        self,
    ) -> None:
        records = [
            _branch(cell, seq=100 + index)
            for index, cell in enumerate(REMOVED_ENVELOPE)
        ]
        shuffled = list(records)
        random.Random(17).shuffle(shuffled)
        first = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        second = coverage_from_branches(
            shuffled,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        self.assertEqual(first.signature, second.signature)
        self.assertEqual(len(first.signature), 64)
        int(first.signature, 16)  # hex digest

    def test_coverage_signature_binds_root_provenance(self) -> None:
        records = [_branch((6, 6), seq=100)]
        first = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V325_ROOT_STATE_SHA,
        )
        second = coverage_from_branches(
            records,
            V325_ROOT_RECORD,
            root_state_signature=V324_PREPUSH_CHECKPOINT_SHA,
        )
        self.assertNotEqual(first.signature, second.signature)

    def test_delta_and_helper_signatures_are_deterministic(self) -> None:
        before = _coverage(
            BASELINE_ENVELOPE,
            root_state_signature=V324_PREPUSH_CHECKPOINT_SHA,
        )
        after = _coverage(
            REMOVED_ENVELOPE, root_state_signature=V325_ROOT_STATE_SHA
        )
        first = delta(
            before, after, excluded_footprint_cells=FOOTPRINT
        )
        second = delta(
            before,
            after,
            excluded_footprint_cells=tuple(reversed(FOOTPRINT)),
        )
        self.assertEqual(first.signature, second.signature)
        self.assertNotEqual(
            first.signature,
            delta(
                before, after, excluded_footprint_cells=()
            ).signature,
        )
        bit = score_target_bit(after, COLUMN8_BAND)
        self.assertEqual(
            bit.signature, score_target_bit(after, COLUMN8_BAND).signature
        )
        agreement = repetition_agreement(after, after)
        self.assertEqual(
            agreement.signature,
            repetition_agreement(after, after).signature,
        )

    def test_endpoint_cell_derivation_matches_probe_analyses(self) -> None:
        self.assertEqual(
            branch_endpoint_cell(V325_CLEAN_BRANCH), (7, 6)
        )
        self.assertEqual(
            branch_endpoint_cell(V325_DEPARTED_BRANCH), (12, 6)
        )
        record = dict(V325_CLEAN_BRANCH)
        del record["human_prior_target_player_slot"]
        self.assertIsNone(branch_endpoint_cell(record))

    def test_event_constant_matches_probe_telemetry(self) -> None:
        self.assertEqual(
            BRANCH_EVENT, "human_prior_option_branch_verified"
        )


if __name__ == "__main__":
    unittest.main()
