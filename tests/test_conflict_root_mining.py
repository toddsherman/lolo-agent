"""Tests for the preregistered WP8 conflict-root mining tool.

Synthetic fixtures exercise the re-scoring, flip detection, family
classification, restorability, determinism, and seeded-design
construction; the real v322-v328 telemetry and the committed manifest are
covered behind existence skips.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lolo_agent.conflict_root_mining import (
    CURRENT_BASELINE,
    CURRENT_MAPPED,
    CURRENT_MISSING,
    DEFAULT_RUN_IDS,
    FAMILY_EXHAUSTION,
    FAMILY_NOVELTY_DECOY,
    FAMILY_POST_EXPLOIT,
    SEEDED_PROVENANCE_MARKER,
    build_manifest,
    build_seeded_record,
    candidate_bonus,
    check_restorability,
    classify_family,
    construct_archive_seeded_design,
    construct_records_variant_designs,
    load_certified_records,
    manifest_digest,
    mine_run,
    record_from_payload,
    resolve_current_record,
    store_from_payloads,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_RECORDS = (
    REPO_ROOT / "experiments/lolo1-wp5/wp8lite-accessibility-records.json"
)
REAL_EVALUATIONS = (
    REPO_ROOT / "experiments/lolo1-entity-v10/evaluations"
)
REAL_MANIFEST = (
    REPO_ROOT / "experiments/lolo1-wp5/conflict-root-manifest.json"
)

BASELINE_CELLS = [
    [6, 6], [6, 7], [6, 8], [6, 9], [6, 10], [7, 10], [8, 10],
]
REMOVAL_CELLS = BASELINE_CELLS + [
    [7, 6], [8, 6], [8, 7], [8, 8], [9, 8], [10, 6], [10, 7], [10, 8],
    [11, 6], [11, 7], [11, 8], [12, 6], [12, 7], [12, 8], [12, 9],
    [12, 10], [12, 11],
]
# The baseline record's (6,7)..(8,10) cells overlap the removal set, so
# the removal-vs-baseline bonus is 17 new cells + 1 milestone x 8 = 25.
ROOT_SENTINEL = "root-sentinel-unmatchable"
REMOVAL_SIGNATURE = "sig-removal"


def _provenance(signature, verification="certified_hold"):
    return {
        "run_id": "synthetic-run",
        "preregistration_doc": "docs/synthetic.md",
        "configuration_signature": signature,
        "verification": verification,
        "certification_predicate": "synthetic hold predicate",
        "certified_branches": 10,
        "total_branches": 10,
        "search_depth": 12,
        "search_beam": 128,
    }


def synthetic_records_payloads():
    return [
        {
            "provenance": _provenance(ROOT_SENTINEL),
            "certified_cells": BASELINE_CELLS,
            "certified_open_frontiers": [],
            "certified_milestone_cells": [],
            "preparation_outcome_category": "none",
            "confirmed_manipulation_count": 0,
            "root_configuration": True,
        },
        {
            "provenance": _provenance(REMOVAL_SIGNATURE),
            "certified_cells": REMOVAL_CELLS,
            "certified_open_frontiers": [],
            "certified_milestone_cells": [[12, 11]],
            "preparation_outcome_category": "removal",
            "confirmed_manipulation_count": 0,
        },
    ]


def synthetic_store():
    return store_from_payloads(synthetic_records_payloads())


def _event(seq, name, **fields):
    payload = {"event": name, "seq": seq}
    payload.update(fields)
    return payload


def _seed_event(seq=1, signature=None):
    return _event(
        seq,
        "human_prior_root_object_state_seeded",
        tracked_world_state_signature=signature,
    )


def _add_event(seq, state_id, score, signature=None, decision=1):
    return _event(
        seq,
        "human_prior_option_archive_added",
        state_id=state_id,
        score=score,
        human_prior_option_tracked_world_state_signature=signature,
        decision=decision,
    )


def _restore_event(seq, state_id, decision, pfv=0.0, **fields):
    return _event(
        seq,
        "archive_branch_restored",
        state_id=state_id,
        decision=decision,
        persistent_frontier_value=pfv,
        reason="human_prior_graph_stagnation",
        **fields,
    )


def _commit_event(seq, decision, slots=None):
    return _event(
        seq,
        "decision_committed",
        decision=decision,
        human_prior_collected_heart_slots=slots,
    )


class _SyntheticRun:
    """Builds a run directory with an events.jsonl and optional states."""

    def __init__(self, root, run_id="synthetic-run"):
        self.run_dir = Path(root) / run_id
        (self.run_dir / "states").mkdir(parents=True)
        self.events = []

    def add(self, event):
        self.events.append(event)
        return self

    def add_snapshot(self, seq, state_id, content=b"state-bytes"):
        digest = hashlib.sha256(content).hexdigest()
        state_file = f"states/{digest}.state"
        (self.run_dir / state_file).write_bytes(content)
        self.events.append(
            _event(
                seq,
                "option_archive_snapshot_stored",
                state_id=state_id,
                state_file=state_file,
                state_sha256=digest,
            )
        )
        return self

    def write(self):
        path = self.run_dir / "events.jsonl"
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in self.events),
            encoding="utf-8",
        )
        return self.run_dir


class RecordStoreTests(unittest.TestCase):
    def test_loads_records_and_root_designation(self):
        store = synthetic_store()
        self.assertEqual(len(store.records), 2)
        self.assertEqual(
            store.root_configuration_signature, ROOT_SENTINEL
        )
        self.assertIn(REMOVAL_SIGNATURE, store.content_signatures())

    def test_duplicate_root_designation_refused(self):
        payloads = synthetic_records_payloads()
        payloads[1]["root_configuration"] = True
        with self.assertRaises(ValueError):
            store_from_payloads(payloads)

    def test_non_boolean_designation_refused(self):
        payloads = synthetic_records_payloads()
        payloads[0]["root_configuration"] = "yes"
        with self.assertRaises(ValueError):
            store_from_payloads(payloads)

    def test_duplicate_configuration_signature_refused(self):
        payloads = synthetic_records_payloads()
        payloads.append(dict(payloads[1]))
        with self.assertRaises(ValueError):
            store_from_payloads(payloads)

    def test_load_from_file_records_sha(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "records.json"
            raw = json.dumps(synthetic_records_payloads()).encode()
            path.write_bytes(raw)
            store = load_certified_records(path)
            self.assertEqual(
                store.file_sha256, hashlib.sha256(raw).hexdigest()
            )


class CurrentResolutionTests(unittest.TestCase):
    def test_empty_signature_resolves_baseline(self):
        record, source = resolve_current_record(synthetic_store(), "")
        self.assertEqual(source, CURRENT_BASELINE)
        self.assertEqual(
            record.provenance.configuration_signature, ROOT_SENTINEL
        )

    def test_mapped_signature_resolves_mapped(self):
        record, source = resolve_current_record(
            synthetic_store(), REMOVAL_SIGNATURE
        )
        self.assertEqual(source, CURRENT_MAPPED)
        self.assertEqual(
            record.provenance.configuration_signature, REMOVAL_SIGNATURE
        )

    def test_unmapped_nonempty_signature_never_falls_back(self):
        record, source = resolve_current_record(
            synthetic_store(), "unknown-signature"
        )
        self.assertIsNone(record)
        self.assertEqual(source, CURRENT_MISSING)

    def test_empty_signature_without_designation_refuses(self):
        payloads = synthetic_records_payloads()
        payloads[0].pop("root_configuration")
        store = store_from_payloads(payloads)
        record, source = resolve_current_record(store, "")
        self.assertIsNone(record)
        self.assertEqual(source, CURRENT_MISSING)


class CandidateBonusTests(unittest.TestCase):
    def test_removal_vs_baseline_scores_25(self):
        store = synthetic_store()
        bonus, refusal = candidate_bonus(
            store, REMOVAL_SIGNATURE, store.root_record
        )
        self.assertEqual(bonus, 25.0)
        self.assertIsNone(refusal)

    def test_predicted_record_scores_zero_with_refusal(self):
        payloads = synthetic_records_payloads()
        payloads[1]["provenance"] = _provenance(
            REMOVAL_SIGNATURE, verification="predicted"
        )
        store = store_from_payloads(payloads)
        bonus, refusal = candidate_bonus(
            store, REMOVAL_SIGNATURE, store.root_record
        )
        self.assertEqual(bonus, 0.0)
        self.assertIsNotNone(refusal)

    def test_unmapped_candidate_scores_zero(self):
        store = synthetic_store()
        bonus, refusal = candidate_bonus(
            store, "unknown", store.root_record
        )
        self.assertEqual(bonus, 0.0)
        self.assertEqual(refusal, "candidate_record_missing")


class FlipDetectionTests(unittest.TestCase):
    def _mine(self, events, store=None):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            for event in events:
                run.add(event)
            run_dir = run.write()
            return mine_run(run_dir, store or synthetic_store())

    def test_certified_baseline_top_is_not_a_conflict(self):
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-A", 50.0, REMOVAL_SIGNATURE),
                _add_event(3, "state-B", 20.0),
                _restore_event(4, "state-A", 2, pfv=50.0),
            ]
        )
        [point] = [
            p for p in result.decision_points if p.kind == "restore"
        ]
        self.assertFalse(point.conflict)
        self.assertEqual(point.baseline_top.state_id, "state-A")
        self.assertEqual(point.combined_top.state_id, "state-A")
        self.assertEqual(result.conflicts, [])

    def test_decoy_over_certified_flips_and_records_margins(self):
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-decoy", 30.0),
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE),
                _restore_event(4, "state-decoy", 2, pfv=30.0),
            ]
        )
        [conflict] = result.conflicts
        point = conflict.point
        self.assertTrue(point.conflict)
        self.assertEqual(point.current_source, CURRENT_BASELINE)
        self.assertEqual(point.baseline_top.state_id, "state-decoy")
        self.assertEqual(point.combined_top.state_id, "state-cert")
        self.assertEqual(point.combined_top.combined_score, 45.0)
        self.assertEqual(conflict.baseline_gap, 10.0)
        self.assertEqual(conflict.minimum_flipping_bonus, 10.0)
        self.assertEqual(conflict.flip_margin, 15.0)
        self.assertEqual(conflict.family, FAMILY_NOVELTY_DECOY)
        self.assertFalse(conflict.instrument_gap_dependent)

    def test_mapped_current_zeroes_bonus_and_conflict(self):
        # After restoring the removal-class branch the root maps to the
        # removal record, so a remaining removal-class candidate scores
        # zero (candidate == current) and no conflict can fire.
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-cert", 40.0, REMOVAL_SIGNATURE),
                _add_event(3, "state-cert2", 20.0, REMOVAL_SIGNATURE),
                _add_event(4, "state-decoy", 30.0),
                _restore_event(5, "state-cert", 2, pfv=40.0),
                _commit_event(6, 2),
                _restore_event(7, "state-decoy", 3, pfv=30.0),
            ]
        )
        late_points = [
            p for p in result.decision_points if p.seq >= 6
        ]
        self.assertTrue(late_points)
        for point in late_points:
            self.assertEqual(point.current_source, CURRENT_MAPPED)
            self.assertEqual(point.positive_bonus_candidates, 0)
            self.assertFalse(point.conflict)

    def test_bonus_cross_check_against_recorded_telemetry(self):
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-cert", 40.0, REMOVAL_SIGNATURE),
                _add_event(3, "state-decoy", 10.0),
                _restore_event(
                    4,
                    "state-cert",
                    2,
                    pfv=65.0,
                    verified_accessibility_bonus=25.0,
                    verified_accessibility_current_source="baseline",
                ),
            ]
        )
        [check] = result.bonus_cross_checks
        self.assertEqual(check.computed_bonus, 25.0)
        self.assertTrue(check.match)
        # The recorded pfv is bonus-inclusive; the harvested baseline
        # value subtracts the recorded bonus.
        [point] = [
            p for p in result.decision_points if p.kind == "restore"
        ]
        self.assertEqual(point.in_run_winner_baseline_value, 40.0)

    def test_deterministic_tie_break_prefers_earlier_add(self):
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-first", 30.0),
                _add_event(3, "state-second", 30.0),
                _restore_event(4, "state-second", 2, pfv=30.0),
            ]
        )
        [point] = [
            p for p in result.decision_points if p.kind == "restore"
        ]
        self.assertEqual(point.baseline_top.state_id, "state-first")

    def test_instrument_gap_reset_flags_later_conflicts(self):
        # A restore of a branch without track metadata resets the root
        # signature to empty; conflicts scored through the resulting
        # baseline fallback are flagged instrument-gap dependent.
        result = self._mine(
            [
                _seed_event(2, REMOVAL_SIGNATURE),
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE),
                _add_event(4, "state-decoy", 30.0),
                _add_event(5, "state-trackless", 25.0),
                _restore_event(6, "state-trackless", 2, pfv=25.0),
                _commit_event(7, 2),
            ]
        )
        [conflict] = result.conflicts
        self.assertEqual(conflict.point.seq, 7)
        self.assertEqual(
            conflict.point.current_source, CURRENT_BASELINE
        )
        self.assertTrue(conflict.instrument_gap_dependent)

    def test_causal_outcome_entries_stay_out_of_the_candidate_set(self):
        result = self._mine(
            [
                _seed_event(),
                _add_event(2, "state-cert", 20.0, REMOVAL_SIGNATURE),
                _event(
                    3,
                    "archive_causal_outcome_added",
                    state_id="state-causal",
                    persistent_frontier_value=99.0,
                    decision=1,
                ),
                _restore_event(
                    4, "state-cert", 2, pfv=20.0, archive_size=1
                ),
            ]
        )
        [point] = [
            p for p in result.decision_points if p.kind == "restore"
        ]
        self.assertEqual(point.candidate_count, 1)
        self.assertEqual(point.recorded_archive_size, 1)
        self.assertEqual(point.baseline_top.state_id, "state-cert")
        # The causal entry is still known for winner joins.
        self.assertIn("state-causal", result.candidates)


class FamilyClassificationTests(unittest.TestCase):
    def test_post_exploit_when_milestones_collected(self):
        store = synthetic_store()
        family = classify_family(
            False, store.records[REMOVAL_SIGNATURE], ((12, 11),)
        )
        self.assertEqual(family, FAMILY_POST_EXPLOIT)

    def test_novelty_decoy_when_milestones_uncollected(self):
        store = synthetic_store()
        family = classify_family(
            False, store.records[REMOVAL_SIGNATURE], ()
        )
        self.assertEqual(family, FAMILY_NOVELTY_DECOY)

    def test_exhaustion_context_wins(self):
        store = synthetic_store()
        family = classify_family(
            True, store.records[REMOVAL_SIGNATURE], ((12, 11),)
        )
        self.assertEqual(family, FAMILY_EXHAUSTION)

    def test_collected_slots_map_to_cells_and_reach_classification(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add(_seed_event())
            run.add(_add_event(2, "state-decoy", 30.0))
            run.add(
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE)
            )
            # Decision 1 commit collects the milestone heart at pixel
            # slot (192, 176) -> cell (12, 11).
            run.add(_commit_event(4, 1, slots=[[192, 176]]))
            run.add(_restore_event(5, "state-decoy", 2, pfv=30.0))
            run_dir = run.write()
            result = mine_run(run_dir, synthetic_store())
        [conflict] = [
            c for c in result.conflicts if c.point.kind == "restore"
        ]
        self.assertEqual(
            conflict.point.collected_milestone_cells, ((12, 11),)
        )
        self.assertEqual(conflict.family, FAMILY_POST_EXPLOIT)

    def test_exhaustion_event_marks_same_decision_points(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add(_seed_event())
            run.add(_add_event(2, "state-decoy", 30.0))
            run.add(
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE)
            )
            run.add(
                _event(
                    4,
                    "goal_milestone_exhaustion_deferred",
                    decision=2,
                )
            )
            run.add(_restore_event(5, "state-decoy", 2, pfv=30.0))
            run_dir = run.write()
            result = mine_run(run_dir, synthetic_store())
        [conflict] = result.conflicts
        self.assertTrue(conflict.point.exhaustion_context)
        self.assertEqual(conflict.family, FAMILY_EXHAUSTION)


class RestorabilityTests(unittest.TestCase):
    def test_verified_when_file_matches_recorded_digest(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add_snapshot(1, "state-A", b"payload")
            run_dir = run.write()
            result = mine_run(run_dir, synthetic_store())
            restorability = check_restorability(
                run_dir, "state-A", result.snapshots
            )
        self.assertTrue(restorability.snapshot_recorded)
        self.assertTrue(restorability.file_exists)
        self.assertTrue(restorability.digest_verified)

    def test_digest_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add_snapshot(1, "state-A", b"payload")
            run_dir = run.write()
            result = mine_run(run_dir, synthetic_store())
            state_file, _sha = result.snapshots["state-A"]
            (run_dir / state_file).write_bytes(b"tampered")
            restorability = check_restorability(
                run_dir, "state-A", result.snapshots
            )
        self.assertTrue(restorability.file_exists)
        self.assertFalse(restorability.digest_verified)

    def test_unrecorded_state_is_not_restorable(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add(_seed_event())
            run_dir = run.write()
            result = mine_run(run_dir, synthetic_store())
            restorability = check_restorability(
                run_dir, "state-missing", result.snapshots
            )
        self.assertFalse(restorability.snapshot_recorded)
        self.assertFalse(restorability.digest_verified)


class SeededDesignTests(unittest.TestCase):
    def _corpus(self, root):
        # Run 1: an unmapped decoy wins a restore at a recorded
        # restore-time value; both its state and the certified branch
        # carry verified snapshots.
        run1 = _SyntheticRun(root, "synthetic-run-1")
        run1.add(_seed_event())
        run1.add_snapshot(2, "state-decoy", b"decoy-state")
        run1.add(_add_event(3, "state-decoy", 12.0))
        run1.add_snapshot(4, "state-cert", b"cert-state")
        run1.add(_add_event(5, "state-cert", 20.0, REMOVAL_SIGNATURE))
        run1.add(_restore_event(6, "state-decoy", 2, pfv=30.0))
        run1_dir = run1.write()
        # Run 2: a restore instant with zero positive-bonus candidates
        # (mapped-record current side is absent entirely) for the
        # records-variant construction.
        run2 = _SyntheticRun(root, "synthetic-run-2")
        run2.add(_seed_event())
        run2.add_snapshot(2, "state-top", b"top-state")
        run2.add(_add_event(3, "state-top", 27.0))
        run2.add_snapshot(4, "state-chal", b"chal-state")
        run2.add(
            _add_event(5, "state-chal", 13.0, "sig-underranked")
        )
        run2.add(_restore_event(6, "state-top", 2, pfv=27.0))
        run2_dir = run2.write()
        store = synthetic_store()
        return [
            mine_run(run1_dir, store),
            mine_run(run2_dir, store),
        ], store

    def test_archive_seeded_design_pairs_decoy_with_certified(self):
        with tempfile.TemporaryDirectory() as root:
            results, store = self._corpus(root)
            design = construct_archive_seeded_design(results, store)
        self.assertTrue(design["constructed"])
        self.assertEqual(design["family"], FAMILY_NOVELTY_DECOY)
        self.assertEqual(design["decoy"]["state_id"], "state-decoy")
        self.assertEqual(
            design["decoy"]["restore_time_baseline_value"], 30.0
        )
        self.assertEqual(
            design["certified_branch"]["state_id"], "state-cert"
        )
        arithmetic = design["arithmetic"]
        self.assertEqual(arithmetic["baseline_gap"], 10.0)
        self.assertEqual(arithmetic["provided_bonus"], 25.0)
        self.assertEqual(arithmetic["flip_margin"], 15.0)
        self.assertTrue(
            design["decoy"]["restorability"]["digest_verified"]
        )
        self.assertTrue(
            design["certified_branch"]["restorability"][
                "digest_verified"
            ]
        )
        self.assertIn("VOID", design["void_condition"])

    def test_archive_seeded_design_requires_a_valid_pair(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add(_seed_event())
            run.add_snapshot(2, "state-cert", b"cert-state")
            run.add(
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE)
            )
            run_dir = run.write()
            store = synthetic_store()
            design = construct_archive_seeded_design(
                [mine_run(run_dir, store)], store
            )
        self.assertFalse(design["constructed"])

    def test_records_variant_seeds_underranked_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            results, store = self._corpus(root)
            variant = construct_records_variant_designs(
                results, store, synthetic_records_payloads()
            )
        self.assertTrue(variant["constructed"])
        designs = variant["valid_designs"]
        self.assertTrue(designs)
        top = designs[0]
        self.assertEqual(top["decoy"]["state_id"], "state-top")
        self.assertEqual(top["challenger"]["state_id"], "state-chal")
        self.assertEqual(
            top["seeded_record"]["provenance"][
                "configuration_signature"
            ],
            "sig-underranked",
        )
        self.assertIn(
            SEEDED_PROVENANCE_MARKER,
            top["seeded_record"]["provenance"]["run_id"],
        )
        self.assertEqual(top["arithmetic"]["baseline_gap"], 14.0)
        self.assertEqual(top["arithmetic"]["provided_bonus"], 25.0)
        self.assertIsNotNone(variant["variant_records_file"])

    def test_seeded_record_marks_construction_and_copies_envelope(self):
        store = synthetic_store()
        seeded = build_seeded_record(
            store, "sig-underranked", "run-x", "state-x"
        )
        self.assertIn(
            SEEDED_PROVENANCE_MARKER, seeded.provenance.run_id
        )
        self.assertIn(
            SEEDED_PROVENANCE_MARKER,
            seeded.provenance.certification_predicate,
        )
        self.assertEqual(
            seeded.certified_cells,
            store.records[REMOVAL_SIGNATURE].certified_cells,
        )


class ManifestTests(unittest.TestCase):
    def _manifest(self, root):
        # The decoy outranks the certified branch on the add-time
        # baseline, so the corpus carries one stageable conflict.
        run = _SyntheticRun(root)
        run.add(_seed_event())
        run.add_snapshot(2, "state-decoy", b"decoy-state")
        run.add(_add_event(3, "state-decoy", 30.0))
        run.add_snapshot(4, "state-cert", b"cert-state")
        run.add(_add_event(5, "state-cert", 20.0, REMOVAL_SIGNATURE))
        run.add(_restore_event(6, "state-decoy", 2, pfv=30.0))
        run_dir = run.write()
        store = synthetic_store()
        results = [mine_run(run_dir, store)]
        return build_manifest(
            results,
            store,
            Path("records.json"),
            Path("evaluations"),
            records_payloads=synthetic_records_payloads(),
        )

    def test_manifest_is_deterministic_and_digest_stable(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "a").mkdir()
            (Path(root) / "b").mkdir()
            first = self._manifest(Path(root) / "a")
            second = self._manifest(Path(root) / "b")
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        self.assertEqual(
            first["digest_sha256"], manifest_digest(first)
        )

    def test_conflicting_corpus_reports_organic_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = self._manifest(root)
        organic = manifest["organic_conflicts"]
        self.assertEqual(organic["total"], 1)
        self.assertEqual(organic["stageable"], 1)
        self.assertEqual(
            organic["by_family"][FAMILY_NOVELTY_DECOY], 1
        )
        # Stageable organic conflicts exist, so no seeded fallback.
        self.assertNotIn("seeded_designs", manifest)

    def test_non_conflicting_corpus_engages_seeded_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            run = _SyntheticRun(root)
            run.add(_seed_event())
            run.add_snapshot(2, "state-cert", b"cert-state")
            run.add(
                _add_event(3, "state-cert", 20.0, REMOVAL_SIGNATURE)
            )
            run.add(_restore_event(4, "state-cert", 2, pfv=20.0))
            run_dir = run.write()
            store = synthetic_store()
            manifest = build_manifest(
                [mine_run(run_dir, store)],
                store,
                Path("records.json"),
                Path(root),
                records_payloads=synthetic_records_payloads(),
            )
        self.assertEqual(manifest["organic_conflicts"]["total"], 0)
        self.assertIn("seeded_designs", manifest)


@unittest.skipUnless(
    REAL_RECORDS.is_file(), "real records file not present"
)
class RealRecordsTests(unittest.TestCase):
    def test_content_signatures_match_run_telemetry(self):
        # The values every v327/v328 run recorded at load time
        # (verified_accessibility_records_loaded, seq 3).
        store = load_certified_records(REAL_RECORDS)
        signatures = store.content_signatures()
        self.assertEqual(
            signatures["85fd9014d58deb42"][:16], "15604cb504868b33"
        )
        self.assertEqual(
            signatures["596a1c8a3c0fc8be"][:16], "37ea410d76472a12"
        )
        self.assertEqual(
            signatures["prepush-root-empty-track-unmatchable"][:16],
            "47975c94dea2b0fe",
        )
        self.assertEqual(
            store.root_configuration_signature,
            "prepush-root-empty-track-unmatchable",
        )


@unittest.skipUnless(
    REAL_RECORDS.is_file()
    and (
        REAL_EVALUATIONS
        / "entity-v328-room3-wp8lite-treatment-w1-d12/events.jsonl"
    ).is_file(),
    "real v328 telemetry not present",
)
class RealTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = load_certified_records(REAL_RECORDS)
        cls.result = mine_run(
            REAL_EVALUATIONS
            / "entity-v328-room3-wp8lite-treatment-w1-d12",
            cls.store,
        )

    def test_recorded_treatment_bonuses_reproduce_exactly(self):
        checks = {
            check.seq: check for check in self.result.bonus_cross_checks
        }
        self.assertEqual(checks[76560].recorded_bonus, 25.0)
        self.assertEqual(checks[76560].computed_bonus, 25.0)
        self.assertEqual(
            checks[76560].computed_current_source, CURRENT_BASELINE
        )
        self.assertTrue(checks[76560].match)
        self.assertTrue(checks[78049].match)
        self.assertTrue(checks[79426].match)
        self.assertEqual(
            checks[78049].computed_current_source, CURRENT_MAPPED
        )

    def test_no_stageable_conflict_in_v328(self):
        stageable = [
            conflict
            for conflict in self.result.conflicts
            if not conflict.instrument_gap_dependent
        ]
        self.assertEqual(stageable, [])

    def test_archive_size_reconciles_at_the_decision_2_instant(self):
        [point] = [
            p
            for p in self.result.decision_points
            if p.kind == "restore" and p.decision == 2
        ]
        self.assertEqual(point.recorded_archive_size, 13)
        self.assertEqual(point.candidate_count, 13)
        self.assertEqual(point.positive_bonus_candidates, 4)
        self.assertEqual(point.baseline_top.state_id, "state-00012257")
        self.assertFalse(point.conflict)


@unittest.skipUnless(
    REAL_MANIFEST.is_file(), "mined manifest not present"
)
class RealManifestTests(unittest.TestCase):
    def test_manifest_digest_is_self_consistent(self):
        manifest = json.loads(REAL_MANIFEST.read_text())
        self.assertEqual(
            manifest["digest_sha256"], manifest_digest(manifest)
        )

    def test_manifest_reports_zero_stageable_and_seeded_designs(self):
        manifest = json.loads(REAL_MANIFEST.read_text())
        self.assertEqual(
            manifest["organic_conflicts"]["stageable"], 0
        )
        self.assertIn("seeded_designs", manifest)
        primary = manifest["seeded_designs"]["primary_archive_seeded"]
        self.assertTrue(primary["constructed"])
        self.assertIn("CONSTRUCTED", primary["disclosure"])
        self.assertEqual(
            set(manifest["corpus"]["run_ids"]), set(DEFAULT_RUN_IDS)
        )


class RecordPayloadRoundTripTests(unittest.TestCase):
    def test_record_from_payload_round_trips_signature(self):
        payloads = synthetic_records_payloads()
        record = record_from_payload(payloads[1])
        again = record_from_payload(payloads[1])
        self.assertEqual(
            record.content_signature(), again.content_signature()
        )


if __name__ == "__main__":
    unittest.main()
