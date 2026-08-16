"""Static strict-lineage linter (direction-review Amendment C).

Answers one question mechanically: could a module's or artifact's
derivation have touched assisted perception? The assisted surface is the
explicitly labelled human-prior code in ``lolo_agent.goal_prior`` — the
player detector, the player pixel mask, and the heart/chest/life template
prototypes. Today the strict/assisted boundary is discipline; this module
turns it into tooling (docs/direction-review-2026-08-16.md §3.C,
docs/roadmap.md §17 item 7).

Two instruments, one entry point:

- an ``ast``-based import/attribute graph walker over a package that
  reports, for an entry module, every transitive chain to the assisted
  surface;
- a checkpoint auditor that checks declared provenance fields in a
  checkpoint JSON payload (anonymous-behavior style) or an already-loaded
  ``.pt`` metadata dict against a strict allowlist.

``lint_strict_lineage(paths)`` combines both into one deterministic,
content-signed report. Every analyzed file is only ever read.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# The module that defines the assisted perception surface. Any transitive
# import of it makes the importer's derivation assisted-coupled.
ASSISTED_MODULE_BASENAME = "goal_prior"

# The named assisted surface: supplied-prior classes, the player detector
# and mask, and the heart/chest/life template prototypes. Referencing any
# of these names — as an import, attribute, parameter, or dynamic-access
# string — is decisive evidence of assisted coupling.
ASSISTED_SYMBOLS = (
    "HEART_PROTOTYPE",
    "HeartGoalAnalysis",
    "OPEN_CHEST_EMPTY_PROTOTYPE",
    "OPEN_CHEST_PROTOTYPE",
    "OPEN_CHEST_PROTOTYPES",
    "PixelHeartGoalPrior",
    "detect_player",
    "player_pixel_mask",
)

# Telemetry emitted by the assisted goal prior is prefixed uniformly.
# Reading such fields is advisory evidence: it proves contact with
# assisted diagnostics but is also how provenance classifiers name the
# marker, so it never flips the verdict by itself.
ASSISTED_TELEMETRY_PREFIX = "human_prior"

REFERENCE_KINDS = (
    "import",
    "definition",
    "symbol",
    "dynamic_string",
    "telemetry_marker",
)
DECISIVE_REFERENCE_KINDS = frozenset(
    {"import", "definition", "symbol", "dynamic_string"}
)

# Checkpoint provenance allowlists. Inputs a strict-lineage artifact may
# declare persistent; everything else — and explicitly every forbidden
# input — is a violation.
STRICT_REWARD_TRACKS = (
    "strict",
    "strict_from_assisted_state",
    "strict_rule_free",
)
STRICT_INPUT_ALLOWLIST = (
    "action_durations",
    "actions",
    "duration_matched_noop_pixels",
    "observed_transition_graph",
    "pixels",
    "save_state_branch_outcomes",
    "save_states",
    "verified_endpoint_pixels",
)
FORBIDDEN_INPUTS = (
    "RAM",
    "evaluator_annotations",
    "level_annotations",
    "object_labels",
    "planner_scores",
    "rewards",
    "solutions",
)

# Case-insensitive substrings that mark a declared field or value as
# assisted-derived. Scanning skips ``excluded_inputs`` values, which
# declare what was *not* used.
ASSISTED_FIELD_MARKERS = (
    "goal_prior",
    "detect_player",
    "player_pixel_mask",
    "human_prior",
    "heart",
    "chest",
    "life_loss",
    "life_counter",
    "life_signature",
)

PROVENANCE_FIELDS = ("reward_track", "persistent_inputs", "excluded_inputs")


def canonical_report_signature(value: Any) -> str:
    """Deterministic, content-derived signature of a JSON payload."""

    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class AssistedReference:
    """One piece of evidence that a module touches the assisted surface."""

    module: str
    kind: str
    symbol: str
    line: int

    @property
    def decisive(self) -> bool:
        return self.kind in DECISIVE_REFERENCE_KINDS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "kind": self.kind,
            "symbol": self.symbol,
            "line": self.line,
            "decisive": self.decisive,
        }


@dataclass(frozen=True)
class ModuleScan:
    """Static facts about one module: imports and assisted references."""

    module: str
    path: str
    imports: Tuple[str, ...]
    references: Tuple[AssistedReference, ...]


class _ModuleVisitor(ast.NodeVisitor):
    """Collect intra-package import edges and assisted references."""

    def __init__(self, package: str, module: str) -> None:
        self.package = package
        self.module = module
        self.is_assisted_module = (
            module.rsplit(".", 1)[-1] == ASSISTED_MODULE_BASENAME
        )
        self.imports: Set[str] = set()
        self.references: Set[AssistedReference] = set()

    def _symbol_kind(self) -> str:
        return "definition" if self.is_assisted_module else "symbol"

    def _add_reference(self, kind: str, symbol: str, line: int) -> None:
        self.references.add(
            AssistedReference(
                module=self.module, kind=kind, symbol=symbol, line=line
            )
        )

    def _record_import(
        self, target: Optional[str], name: Optional[str], line: int
    ) -> None:
        if target is not None:
            self.imports.add(target)
        target_basename = (
            target.rsplit(".", 1)[-1] if target is not None else ""
        )
        assisted_target = target_basename == ASSISTED_MODULE_BASENAME
        assisted_name = name in ASSISTED_SYMBOLS
        if not assisted_target and not assisted_name:
            return
        if target is not None and name is not None:
            symbol = f"{target}.{name}"
        elif target is not None:
            symbol = target
        else:
            symbol = str(name)
        self._add_reference("import", symbol, line)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            dotted = alias.name
            if dotted == self.package or dotted.startswith(
                self.package + "."
            ):
                self._record_import(dotted, None, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source: Optional[str] = None
        if node.level > 0:
            source = (
                self.package
                if node.module is None
                else f"{self.package}.{node.module}"
            )
        elif node.module is not None and (
            node.module == self.package
            or node.module.startswith(self.package + ".")
        ):
            source = node.module
        if source is None:
            for alias in node.names:
                if alias.name in ASSISTED_SYMBOLS:
                    self._record_import(None, alias.name, node.lineno)
        elif source == self.package:
            # ``from . import goal_prior`` / ``from lolo_agent import X``:
            # each alias may be a submodule; the graph keeps only names
            # that exist. An absolute package import also depends on the
            # package ``__init__`` re-exports.
            if node.level == 0:
                self._record_import(source, None, node.lineno)
            for alias in node.names:
                self._record_import(
                    f"{self.package}.{alias.name}", None, node.lineno
                )
        else:
            for alias in node.names:
                self._record_import(source, alias.name, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ASSISTED_SYMBOLS:
            self._add_reference(self._symbol_kind(), node.id, node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ASSISTED_SYMBOLS:
            self._add_reference(self._symbol_kind(), node.attr, node.lineno)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg in ASSISTED_SYMBOLS:
            self._add_reference(self._symbol_kind(), node.arg, node.lineno)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg in ASSISTED_SYMBOLS:
            line = getattr(node, "lineno", getattr(node.value, "lineno", 0))
            self._add_reference(self._symbol_kind(), str(node.arg), line)
        self.generic_visit(node)

    def _visit_definition(self, node: Any) -> None:
        if node.name in ASSISTED_SYMBOLS:
            self._add_reference(self._symbol_kind(), node.name, node.lineno)
        self.generic_visit(node)

    visit_FunctionDef = _visit_definition
    visit_AsyncFunctionDef = _visit_definition
    visit_ClassDef = _visit_definition

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            if node.value in ASSISTED_SYMBOLS:
                self._add_reference(
                    "dynamic_string", node.value, node.lineno
                )
            elif node.value.startswith(ASSISTED_TELEMETRY_PREFIX):
                self._add_reference(
                    "telemetry_marker", node.value, node.lineno
                )
        self.generic_visit(node)


def scan_module_source(source: str, package: str, module: str, path: str) -> ModuleScan:
    """Scan one module's source text without importing or executing it."""

    visitor = _ModuleVisitor(package, module)
    visitor.visit(ast.parse(source, filename=path))
    return ModuleScan(
        module=module,
        path=path,
        imports=tuple(sorted(visitor.imports)),
        references=tuple(
            sorted(
                visitor.references,
                key=lambda ref: (ref.line, ref.kind, ref.symbol),
            )
        ),
    )


@dataclass(frozen=True)
class PackageGraph:
    """Import/attribute graph over every top-level module of a package."""

    package: str
    root: str
    scans: Tuple[ModuleScan, ...]

    @classmethod
    def scan(cls, package_root: Path) -> "PackageGraph":
        root = Path(package_root).expanduser().resolve()
        if not (root / "__init__.py").is_file():
            raise ValueError(
                f"{root} is not a package: missing __init__.py"
            )
        package = root.name
        scans = []
        for source_path in sorted(root.glob("*.py")):
            stem = source_path.stem
            module = package if stem == "__init__" else f"{package}.{stem}"
            scans.append(
                scan_module_source(
                    source_path.read_text(encoding="utf-8"),
                    package,
                    module,
                    str(source_path),
                )
            )
        return cls(
            package=package,
            root=str(root),
            scans=tuple(sorted(scans, key=lambda scan: scan.module)),
        )

    def scan_for(self, module: str) -> ModuleScan:
        for scan in self.scans:
            if scan.module == module:
                return scan
        raise ValueError(f"unknown module: {module}")

    def module_names(self) -> Tuple[str, ...]:
        return tuple(scan.module for scan in self.scans)


@dataclass(frozen=True)
class LineageFinding:
    """One assisted reference plus the exact module chain reaching it."""

    chain: Tuple[str, ...]
    reference: AssistedReference

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain": list(self.chain),
            "reference": self.reference.to_dict(),
        }


@dataclass(frozen=True)
class ModuleLineage:
    """The linted verdict for one entry module."""

    module: str
    path: str
    assisted: bool
    findings: Tuple[LineageFinding, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "path": self.path,
            "assisted": self.assisted,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def trace_module_lineage(
    entry_module: str, graph: PackageGraph
) -> ModuleLineage:
    """Breadth-first transitive trace from one entry module.

    Chains are shortest import paths; the package ``__init__`` is
    traversed only when explicitly imported, so incidental package
    initialization does not launder lineage.
    """

    entry = graph.scan_for(entry_module)
    known = set(graph.module_names())
    chains: Dict[str, Tuple[str, ...]] = {
        entry.module: (entry.module,)
    }
    queue: List[str] = [entry.module]
    while queue:
        current = queue.pop(0)
        for target in graph.scan_for(current).imports:
            if target in known and target not in chains:
                chains[target] = chains[current] + (target,)
                queue.append(target)
    findings = []
    for module, chain in chains.items():
        for reference in graph.scan_for(module).references:
            findings.append(
                LineageFinding(chain=chain, reference=reference)
            )
    findings.sort(
        key=lambda finding: (
            finding.chain,
            finding.reference.line,
            finding.reference.kind,
            finding.reference.symbol,
        )
    )
    return ModuleLineage(
        module=entry.module,
        path=entry.path,
        assisted=any(
            finding.reference.decisive for finding in findings
        ),
        findings=tuple(findings),
    )


@dataclass(frozen=True)
class ProvenanceViolation:
    """One declared-provenance rule breach inside a checkpoint payload."""

    location: str
    reason: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "reason": self.reason,
            "value": self.value,
        }


@dataclass(frozen=True)
class CheckpointAudit:
    """Declared-provenance audit result for one checkpoint payload."""

    label: str
    declared_fields: Tuple[str, ...]
    violations: Tuple[ProvenanceViolation, ...]

    @property
    def assisted(self) -> bool:
        return bool(self.violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "declared_fields": list(self.declared_fields),
            "violations": [
                violation.to_dict() for violation in self.violations
            ],
            "assisted": self.assisted,
        }


def _marker_matches(text: str) -> Tuple[str, ...]:
    lowered = text.lower()
    return tuple(
        marker for marker in ASSISTED_FIELD_MARKERS if marker in lowered
    )


def _scan_markers(
    value: Any,
    location: str,
    violations: List[ProvenanceViolation],
    seen: Set[int],
) -> None:
    if isinstance(value, str):
        matched = _marker_matches(value)
        if matched:
            violations.append(
                ProvenanceViolation(
                    location=location,
                    reason=(
                        "assisted marker in value: "
                        + ", ".join(matched)
                    ),
                    value=value,
                )
            )
        return
    if isinstance(value, Mapping):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key in sorted(value, key=str):
            key_text = str(key)
            key_location = (
                key_text if not location else f"{location}.{key_text}"
            )
            matched = _marker_matches(key_text)
            if matched:
                violations.append(
                    ProvenanceViolation(
                        location=key_location,
                        reason=(
                            "assisted marker in field name: "
                            + ", ".join(matched)
                        ),
                        value=key_text,
                    )
                )
            if key_text == "excluded_inputs":
                # Exclusion declarations describe what was *not* used.
                continue
            _scan_markers(value[key], key_location, violations, seen)
        return
    if isinstance(value, (list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        for index, item in enumerate(value):
            _scan_markers(
                item, f"{location}[{index}]", violations, seen
            )


def audit_checkpoint_metadata(
    payload: Mapping[str, Any], label: str
) -> CheckpointAudit:
    """Audit declared provenance in a checkpoint metadata mapping.

    Accepts a parsed anonymous-behavior-style checkpoint JSON object or
    the metadata dict loaded from a ``.pt`` checkpoint (weights removed or
    present — tensors are ignored by the scan).
    """

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    violations: List[ProvenanceViolation] = []
    declared = tuple(
        field for field in PROVENANCE_FIELDS if field in payload
    )
    reward_track = payload.get("reward_track")
    if "reward_track" in payload:
        if not isinstance(reward_track, str):
            violations.append(
                ProvenanceViolation(
                    location="reward_track",
                    reason="reward_track must be a string",
                    value=repr(reward_track),
                )
            )
        elif reward_track not in STRICT_REWARD_TRACKS:
            violations.append(
                ProvenanceViolation(
                    location="reward_track",
                    reason=(
                        "reward track outside the strict allowlist "
                        f"{sorted(STRICT_REWARD_TRACKS)}"
                    ),
                    value=reward_track,
                )
            )
    persistent = payload.get("persistent_inputs")
    if "persistent_inputs" in payload:
        if not isinstance(persistent, (list, tuple)):
            violations.append(
                ProvenanceViolation(
                    location="persistent_inputs",
                    reason="persistent_inputs must be a list of strings",
                    value=repr(persistent),
                )
            )
        else:
            for index, item in enumerate(persistent):
                location = f"persistent_inputs[{index}]"
                if not isinstance(item, str):
                    violations.append(
                        ProvenanceViolation(
                            location=location,
                            reason="persistent input must be a string",
                            value=repr(item),
                        )
                    )
                elif item in FORBIDDEN_INPUTS:
                    violations.append(
                        ProvenanceViolation(
                            location=location,
                            reason="forbidden input declared persistent",
                            value=item,
                        )
                    )
                elif item not in STRICT_INPUT_ALLOWLIST:
                    violations.append(
                        ProvenanceViolation(
                            location=location,
                            reason=(
                                "input outside the strict allowlist "
                                f"{sorted(STRICT_INPUT_ALLOWLIST)}"
                            ),
                            value=item,
                        )
                    )
    excluded = payload.get("excluded_inputs")
    if "excluded_inputs" in payload:
        if not isinstance(excluded, (list, tuple)):
            violations.append(
                ProvenanceViolation(
                    location="excluded_inputs",
                    reason="excluded_inputs must be a list of strings",
                    value=repr(excluded),
                )
            )
        else:
            for index, item in enumerate(excluded):
                if not isinstance(item, str):
                    violations.append(
                        ProvenanceViolation(
                            location=f"excluded_inputs[{index}]",
                            reason="excluded input must be a string",
                            value=repr(item),
                        )
                    )
    _scan_markers(payload, "", violations, set())
    violations.sort(
        key=lambda violation: (
            violation.location,
            violation.reason,
            violation.value,
        )
    )
    return CheckpointAudit(
        label=label,
        declared_fields=declared,
        violations=tuple(violations),
    )


def audit_checkpoint_json(path: Path) -> CheckpointAudit:
    """Audit a checkpoint JSON file (anonymous-behavior style) in place."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"checkpoint JSON must contain an object: {resolved}"
        )
    return audit_checkpoint_metadata(payload, label=str(resolved))


@dataclass(frozen=True)
class StrictLineageReport:
    """Deterministic combined verdict for a set of lint targets."""

    modules: Tuple[ModuleLineage, ...]
    checkpoints: Tuple[CheckpointAudit, ...]

    @property
    def assisted(self) -> bool:
        return any(entry.assisted for entry in self.modules) or any(
            entry.assisted for entry in self.checkpoints
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modules": [entry.to_dict() for entry in self.modules],
            "checkpoints": [
                entry.to_dict() for entry in self.checkpoints
            ],
            "assisted": self.assisted,
        }

    @property
    def report_signature(self) -> str:
        return canonical_report_signature(self.to_dict())

    def to_event(self) -> Dict[str, Any]:
        """Telemetry payload for `strict_lineage_linted`."""

        return {
            "event": "strict_lineage_linted",
            "assisted_lineage_detected": self.assisted,
            "assisted_modules": [
                entry.module for entry in self.modules if entry.assisted
            ],
            "module_count": len(self.modules),
            "checkpoint_count": len(self.checkpoints),
            "checkpoint_violation_count": sum(
                len(entry.violations) for entry in self.checkpoints
            ),
            "report_signature": self.report_signature,
        }


def _package_root_for(module_path: Path) -> Path:
    directory = module_path.parent
    if not (directory / "__init__.py").is_file():
        raise ValueError(
            f"{module_path} is not inside a package: missing __init__.py"
        )
    while (directory.parent / "__init__.py").is_file():
        directory = directory.parent
    return directory


def lint_strict_lineage(
    paths: Sequence[Path],
    package_root: Optional[Path] = None,
) -> StrictLineageReport:
    """Lint modules (`.py`) and checkpoint JSON files (`.json`).

    `.pt` payloads cannot be decoded with the standard library; audit
    their already-loaded metadata dicts via `audit_checkpoint_metadata`.
    The result is deterministically ordered regardless of input order.
    """

    graphs: Dict[str, PackageGraph] = {}
    if package_root is not None:
        graph = PackageGraph.scan(Path(package_root))
        graphs[graph.root] = graph
    modules: List[ModuleLineage] = []
    checkpoints: List[CheckpointAudit] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved.suffix == ".py":
            root = (
                Path(next(iter(graphs)))
                if package_root is not None
                else _package_root_for(resolved)
            )
            key = str(root)
            if key not in graphs:
                graphs[key] = PackageGraph.scan(root)
            graph = graphs[key]
            stem = resolved.stem
            module = (
                graph.package
                if stem == "__init__"
                else f"{graph.package}.{stem}"
            )
            modules.append(trace_module_lineage(module, graph))
        elif resolved.suffix == ".json":
            checkpoints.append(audit_checkpoint_json(resolved))
        else:
            raise ValueError(
                f"unsupported lint target {resolved}: pass .py modules or "
                ".json checkpoints; audit .pt metadata dicts via "
                "audit_checkpoint_metadata"
            )
    return StrictLineageReport(
        modules=tuple(
            sorted(modules, key=lambda entry: (entry.module, entry.path))
        ),
        checkpoints=tuple(
            sorted(checkpoints, key=lambda entry: entry.label)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether module or checkpoint derivations could have "
            "touched assisted perception"
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = lint_strict_lineage(args.paths, package_root=args.package_root)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 1 if report.assisted else 0


if __name__ == "__main__":
    raise SystemExit(main())
