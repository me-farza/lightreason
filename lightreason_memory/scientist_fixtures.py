"""Synthetic, non-sensitive materials-science and calibration workflow fixtures.

Provides deterministic observations, ordered protocols, humidity exceptions,
blank/reference controls, measured outcomes, versioned procedural corrections,
transient lab chatter, unseen held-out workflow variations, and rubric scorers.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from .store import Store


@dataclass(frozen=True)
class ObservationSpec:
    """One timestamped lab event for durable neural state and/or evidence store."""

    event_id: str
    slot_key: str
    compact_value: str
    content: str
    source: str
    timestamp: str
    kind: str
    role: str  # "durable_fact", "procedure", "measurement", "transient"
    is_transient: bool = False


@dataclass(frozen=True)
class CorrectionSpec:
    """Supersedes an earlier observation in both versioned Store and neural bank."""

    target_event_id: str
    slot_key: str
    previous_compact_value: str
    compact_value: str
    replacement: str
    source: str
    timestamp: str


@dataclass(frozen=True)
class QueryRubric:
    """Deterministic evaluation query and regex rubric for reproducible scoring."""

    name: str
    route: str  # "memory_only", "explicit_lookup", "both", "passthrough_miss"
    question: str
    memory_query: str
    expected_patterns: dict[str, str]
    forbidden_patterns: dict[str, str]
    as_of: str | None = None
    requires_citations: bool = False
    abstain: bool = False
    evidence_context_pattern: str | None = None


@dataclass(frozen=True)
class WorkflowFixture:
    """Complete multi-session scientific workflow with unseen-variation parity."""

    workflow_id: str
    title: str
    domain: str
    initial_as_of: str
    corrected_as_of: str
    observations: tuple[ObservationSpec, ...]
    correction: CorrectionSpec
    queries: tuple[QueryRubric, ...]


ZNO_WORKFLOW = WorkflowFixture(
    workflow_id="zno-spin-xrd-v1",
    title="ZnO Sol-Gel Spin-Coating & XRD Goniometer Calibration",
    domain="materials_calibration",
    initial_as_of="2026-11-10T18:00:00Z",
    corrected_as_of="2026-11-12T12:00:00Z",
    observations=(
        ObservationSpec(
            event_id="zno-design-01",
            slot_key="precursor",
            compact_value="0.20M silica",
            content=(
                "Experiment design choice: compare 0.20 M versus 0.35 M zinc acetate dihydrate "
                "in 2-methoxyethanol (1:1 monoethanolamine stabilizer) on UV-ozone cleaned "
                "fused-silica substrates, prioritizing 0.20 M silica for low-defect optical films."
            ),
            source="lab-notebook:zno-2026-p14",
            timestamp="2026-11-10T09:00:00Z",
            kind="user_report",
            role="durable_fact",
        ),
        ObservationSpec(
            event_id="zno-controls-01",
            slot_key="controls",
            compact_value="blank SRM640f",
            content=(
                "Required controls: include one uncoated fused-silica blank coupon in every "
                "spin/anneal batch for optical background subtraction, and verify the XRD "
                "goniometer against NIST SRM 640f silicon Si(111) at 28.443 deg 2-theta "
                "within +/-0.015 deg before sample scans."
            ),
            source="sop:xrd-cal-v3#sec2",
            timestamp="2026-11-10T09:15:00Z",
            kind="tool_observation",
            role="procedure",
        ),
        ObservationSpec(
            event_id="zno-protocol-01",
            slot_key="anneal",
            compact_value="350C 10C/min",
            content=(
                "Ordered deposition protocol: Step 1 verify NIST SRM 640f Si(111) peak at 28.443 deg "
                "(+/-0.015 deg). Step 2 UV-ozone clean fused-silica coupons for 10 min alongside one "
                "uncoated silica blank. Step 3 dispense 80 uL precursor and spin at 3000 rpm for 30 s "
                "(500 rpm/s ramp). Step 4 soft-bake at 150 C for 5 min, then tube-furnace anneal in air "
                "at 350 C for 60 min using a 10 C/min ramp. Step 5 measure grazing-incidence XRD "
                "(25 to 60 deg 2-theta, 0.02 deg step) and four-point sheet resistance."
            ),
            source="sop:zno-spin-v1#steps1-5",
            timestamp="2026-11-10T09:30:00Z",
            kind="user_report",
            role="procedure",
        ),
        ObservationSpec(
            event_id="zno-exception-01",
            slot_key="humidity",
            compact_value="nitrogen 120s",
            content=(
                "Humidity exception rule: whenever ambient lab relative humidity exceeds 55% RH, "
                "purge the spin-coater bowl with dry nitrogen for 120 s before dispensing and keep "
                "transfer to the 150 C hotplate under 15 s to prevent hydrolysis haze."
            ),
            source="sop:zno-spin-v1#exception-rh",
            timestamp="2026-11-10T09:45:00Z",
            kind="user_report",
            role="durable_fact",
        ),
        ObservationSpec(
            event_id="zno-outcome-01",
            slot_key="outcome",
            compact_value="pinholes 14/cm2",
            content=(
                "Measured batch outcomes: Run ZNO-24-A (0.20 M, 350 C at 10 C/min, 42% RH) gave "
                "118 nm film thickness and ZnO(002) FWHM 0.34 deg, but optical microscopy counted "
                "14 pinholes/cm2 from rapid solvent burnout. Run ZNO-24-B (0.35 M, 58% RH without "
                "nitrogen purge) showed milky hydrolysis haze and FWHM widened to 0.51 deg."
            ),
            source="instrument:xrd-profilometer-run24",
            timestamp="2026-11-10T16:20:00Z",
            kind="tool_observation",
            role="measurement",
        ),
        ObservationSpec(
            event_id="zno-transient-01",
            slot_key="cart",
            compact_value="hallway",
            content=(
                "Temporary note for this afternoon only: the shared multimeter cart is parked "
                "near the hallway printer until 5 pm."
            ),
            source="chat:lab-slack-ephemeral",
            timestamp="2026-11-10T16:45:00Z",
            kind="user_report",
            role="transient",
            is_transient=True,
        ),
    ),
    correction=CorrectionSpec(
        target_event_id="zno-protocol-01",
        slot_key="anneal",
        previous_compact_value="350C 10C/min",
        compact_value="425C 3C/min",
        replacement=(
            "Ordered deposition protocol (rev 2): Step 1 verify NIST SRM 640f Si(111) peak at "
            "28.443 deg (+/-0.015 deg). Step 2 UV-ozone clean fused-silica coupons for 10 min "
            "alongside one uncoated silica blank. Step 3 dispense 80 uL precursor and spin at "
            "3000 rpm for 30 s (500 rpm/s ramp). Step 4 soft-bake at 150 C for 5 min, then "
            "tube-furnace anneal in air at 425 C for 45 min using a 3 C/min ramp to eliminate "
            "pinholes. Step 5 measure grazing-incidence XRD (25 to 60 deg 2-theta, 0.02 deg step) "
            "and four-point sheet resistance."
        ),
        source="sop:zno-spin-v2#rev2-pinhole-fix",
        timestamp="2026-11-11T14:00:00Z",
    ),
    queries=(
        QueryRubric(
            name="propose_initial",
            route="memory_only",
            question=(
                "Propose our next low-defect ZnO fused-silica film run when lab humidity is 60% RH, "
                "including precursor concentration, furnace anneal schedule, atmosphere precaution, "
                "and required controls, with a brief rationale."
            ),
            memory_query="precursor anneal humidity controls for ZnO silica run",
            as_of="2026-11-10T18:00:00Z",
            expected_patterns={
                "0.20 M precursor": r"\b0\.20\s*M\b|\b0\.2\s*M\b",
                "initial 350 C anneal": r"\b350\s*°?\s*C\b",
                "dry nitrogen purge exception": r"\bnitrogen\b|\bN2\b",
                "blank or SRM640f control": r"\bblank\b|\b640f\b",
            },
            forbidden_patterns={
                "future 425 C leaked early": r"\b425\s*°?\s*C\b",
                "transient hallway cart leaked": r"\bhallway printer\b|\bmultimeter cart\b",
            },
        ),
        QueryRubric(
            name="propose_after_correction_paraphrase",
            route="memory_only",
            question=(
                "We are queuing tomorrow's ZnO deposition on fused silica with room humidity at 62% RH. "
                "What thermal ramp/anneal target, spin-bowl atmosphere step, and calibration/blank checks "
                "should we run?"
            ),
            memory_query="anneal humidity controls precursor for ZnO silica deposition",
            expected_patterns={
                "corrected 425 C target": r"\b425\s*°?\s*C\b",
                "corrected 3 C/min ramp": r"\b3\s*°?\s*C\s*/\s*min\b|\b3\s*C/min\b",
                "dry nitrogen purge": r"\bnitrogen\b|\bN2\b",
                "controls retained": r"\bblank\b|\b640f\b",
            },
            forbidden_patterns={
                "stale 350 C recommended as current": r"(?:anneal|run|target|heat)[^.\n]{0,40}\b350\s*°?\s*C\b",
                "transient hallway cart leaked": r"\bhallway printer\b|\bmultimeter cart\b",
            },
        ),
        QueryRubric(
            name="explicit_lookup_current",
            route="explicit_lookup",
            question=(
                "Quote the exact current ZnO spin-coating ordered steps, the Si(111) calibration angle "
                "and tolerance, dispense volume, and updated furnace temperature and ramp rate."
            ),
            memory_query="anneal controls",
            expected_patterns={
                "Si(111) 28.443 deg": r"\b28\.443\b",
                "tolerance 0.015 deg": r"\b0\.015\b",
                "dispense 80 uL": r"\b80\s*uL\b|\b80\s*µL\b",
                "spin 3000 rpm": r"\b3000\s*rpm\b",
                "updated 425 C": r"\b425\s*°?\s*C\b",
                "updated 3 C/min": r"\b3\s*°?\s*C\s*/\s*min\b|\b3\s*C/min\b",
            },
            forbidden_patterns={
                "transient hallway cart leaked": r"\bhallway printer\b",
            },
            requires_citations=True,
        ),
        QueryRubric(
            name="explicit_lookup_historical",
            route="explicit_lookup",
            question="Before the pinhole revision, what furnace anneal temperature and ramp rate were in Step 4?",
            memory_query="anneal",
            as_of="2026-11-10T18:00:00Z",
            expected_patterns={
                "historical 350 C": r"\b350\s*°?\s*C\b",
                "historical 10 C/min": r"\b10\s*°?\s*C\s*/\s*min\b|\b10\s*C/min\b",
            },
            forbidden_patterns={
                "future 425 C in historical view": r"\b425\s*°?\s*C\b",
            },
            requires_citations=True,
            evidence_context_pattern=r"\bStep 4\b[\s\S]*?(?=\bStep 5\b|$)",
        ),
        QueryRubric(
            name="absent_hall_mobility",
            route="explicit_lookup",
            question="What Hall-effect electron mobility in cm^2/V-s was measured for Run ZNO-24-A?",
            memory_query="Hall mobility ZNO-24-A",
            expected_patterns={
                "abstains cleanly": r"\b(?:unknown|not recorded|not measured|unavailable|no Hall|insufficient|cannot)\b",
            },
            forbidden_patterns={
                "fabricated mobility value": r"\b\d+(?:\.\d+)?\s*cm\^?2\s*/\s*V",
            },
            abstain=True,
        ),
    ),
)


TIO2_UNSEEN_VARIATION = WorkflowFixture(
    workflow_id="tio2-dip-raman-v1",
    title="TiO2 Dip-Coated Anatase Film & 532 nm Raman Calibration (Held-Out Variation)",
    domain="materials_calibration_heldout",
    initial_as_of="2026-11-12T18:00:00Z",
    corrected_as_of="2026-11-13T18:00:00Z",
    observations=(
        ObservationSpec(
            event_id="tio2-design-01",
            slot_key="sol",
            compact_value="0.15M quartz",
            content=(
                "Experiment design choice: compare 0.15 M versus 0.30 M titanium isopropoxide sol "
                "(acetylacetone chelate in ethanol) on plasma-cleaned quartz slides, selecting "
                "0.15 M quartz for crack-free anatase waveguides."
            ),
            source="lab-notebook:tio2-2026-p03",
            timestamp="2026-11-12T10:00:00Z",
            kind="user_report",
            role="durable_fact",
        ),
        ObservationSpec(
            event_id="tio2-controls-01",
            slot_key="reference",
            compact_value="bare Si520.7",
            content=(
                "Required controls: process one bare quartz witness slide per dip batch and verify "
                "the 532 nm Raman spectrometer on crystalline silicon at 520.7 cm^-1 (+/-0.3 cm^-1) "
                "at <= 2.0 mW laser power before film acquisition."
            ),
            source="sop:raman-532-cal#sec1",
            timestamp="2026-11-12T10:15:00Z",
            kind="tool_observation",
            role="procedure",
        ),
        ObservationSpec(
            event_id="tio2-protocol-01",
            slot_key="withdraw",
            compact_value="2.0mm/s 480C",
            content=(
                "Ordered dip-coating protocol: Step 1 check Raman Si line at 520.7 cm^-1 (+/-0.3 cm^-1, "
                "<= 2.0 mW). Step 2 O2-plasma clean quartz slides for 8 min with one bare quartz witness. "
                "Step 3 immerse 60 s and withdraw vertically at 2.0 mm/s. Step 4 dry at 120 C for 10 min, "
                "then calcine in dry air at 480 C for 40 min (5 C/min ramp). Step 5 record 532 nm Raman "
                "anatase Eg mode (144 cm^-1) and spectroscopic ellipsometry thickness."
            ),
            source="sop:tio2-dip-v1#steps1-5",
            timestamp="2026-11-12T10:30:00Z",
            kind="user_report",
            role="procedure",
        ),
        ObservationSpec(
            event_id="tio2-exception-01",
            slot_key="enclosure",
            compact_value="argon 90s",
            content=(
                "Humidity exception rule: if enclosure relative humidity exceeds 50% RH, flush the "
                "dip chamber with dry argon for 90 s prior to immersion to block TiO2 precipitation."
            ),
            source="sop:tio2-dip-v1#exception-ar",
            timestamp="2026-11-12T10:45:00Z",
            kind="user_report",
            role="durable_fact",
        ),
        ObservationSpec(
            event_id="tio2-outcome-01",
            slot_key="ellipsometry",
            compact_value="streaks 165nm",
            content=(
                "Measured batch outcomes: Run TIO2-07-A (0.15 M, 2.0 mm/s withdrawal, 480 C, 38% RH) "
                "yielded 165 nm thickness with edge drainage streaks and stress micro-cracks; anatase "
                "Eg peak appeared at 144.2 cm^-1."
            ),
            source="instrument:raman-ellipsometer-run07",
            timestamp="2026-11-12T15:30:00Z",
            kind="tool_observation",
            role="measurement",
        ),
        ObservationSpec(
            event_id="tio2-transient-01",
            slot_key="courier",
            compact_value="lobby",
            content=(
                "Temporary note for today only: optical mount boxes are waiting at the lobby desk "
                "for afternoon pickup."
            ),
            source="chat:lab-slack-ephemeral",
            timestamp="2026-11-12T16:00:00Z",
            kind="user_report",
            role="transient",
            is_transient=True,
        ),
    ),
    correction=CorrectionSpec(
        target_event_id="tio2-protocol-01",
        slot_key="withdraw",
        previous_compact_value="2.0mm/s 480C",
        compact_value="0.8mm/s 500C",
        replacement=(
            "Ordered dip-coating protocol (rev 2): Step 1 check Raman Si line at 520.7 cm^-1 "
            "(+/-0.3 cm^-1, <= 2.0 mW). Step 2 O2-plasma clean quartz slides for 8 min with one "
            "bare quartz witness. Step 3 immerse 60 s and withdraw vertically at 0.8 mm/s for "
            "thinner crack-free drainage. Step 4 dry at 120 C for 10 min, then calcine in dry air "
            "at 500 C for 45 min (2 C/min ramp). Step 5 record 532 nm Raman anatase Eg mode "
            "(144 cm^-1) and spectroscopic ellipsometry thickness."
        ),
        source="sop:tio2-dip-v2#rev2-streak-fix",
        timestamp="2026-11-13T11:00:00Z",
    ),
    queries=(
        QueryRubric(
            name="unseen_after_correction_paraphrase",
            route="memory_only",
            question=(
                "We are coating quartz waveguides with TiO2 sol today while room humidity is 56% RH. "
                "What withdrawal speed, calcination target, chamber gas precaution, and calibration/witness "
                "controls should we use?"
            ),
            memory_query="withdraw enclosure reference sol for TiO2 quartz coating",
            expected_patterns={
                "updated 0.8 mm/s withdrawal": r"\b0\.8\s*mm\s*/\s*s\b",
                "updated 500 C calcination": r"\b500\s*°?\s*C\b",
                "dry argon flush": r"\bargon\b",
                "silicon 520.7 or witness control": r"\b520\.7\b|\bwitness\b|\bbare quartz\b",
            },
            forbidden_patterns={
                "stale 2.0 mm/s withdrawal recommended": r"(?:withdraw|pull|speed)[^.\n]{0,40}\b2\.0\s*mm\s*/\s*s\b",
                "transient lobby boxes leaked": r"\blobby desk\b",
            },
        ),
        QueryRubric(
            name="unseen_explicit_lookup",
            route="explicit_lookup",
            question=(
                "Give the exact current TiO2 dip-coating Raman Si calibration wavenumber, tolerance, "
                "max laser power, and updated withdrawal rate."
            ),
            memory_query="reference withdraw",
            expected_patterns={
                "520.7 cm^-1": r"\b520\.7\b",
                "0.3 cm^-1 tolerance": r"\b0\.3\b",
                "2.0 mW laser": r"\b2(?:\.0)?\s*mW\b",
                "0.8 mm/s withdrawal": r"\b0\.8\s*mm\s*/\s*s\b",
            },
            forbidden_patterns={
                "lobby desk leaked": r"\blobby desk\b",
            },
            requires_citations=True,
        ),
    ),
)


UNRELATED_CONTROL_OBSERVATIONS = (
    ObservationSpec(
        event_id="visc-unrelated-01",
        slot_key="viscometer",
        compact_value="25.0C glycerol",
        content=(
            "Calibration note: equilibrate the capillary viscometer water bath at 25.0 C (+/-0.02 C) "
            "for 20 min using aqueous glycerol standard lot GL-88 before efflux timing."
        ),
        source="sop:viscometer-bath-v1",
        timestamp="2026-11-09T08:30:00Z",
        kind="tool_observation",
        role="procedure",
    ),
)

WORKFLOWS = {
    ZNO_WORKFLOW.workflow_id: ZNO_WORKFLOW,
    TIO2_UNSEEN_VARIATION.workflow_id: TIO2_UNSEEN_VARIATION,
}


def seed_workflow_store(store: Store, fixture: WorkflowFixture = ZNO_WORKFLOW, *, apply_correction: bool = True) -> dict:
    """Populate a scoped SQLite Store with workflow observations and optional correction."""
    by_event = {}
    for obs in fixture.observations:
        if obs.is_transient:
            continue
        by_event[obs.event_id] = store.observe(
            obs.event_id, obs.content, obs.source, obs.timestamp, obs.kind,
        )
    corrected = None
    if apply_correction:
        target = by_event[fixture.correction.target_event_id]
        corrected = store.correct(
            target["memory_id"],
            fixture.correction.replacement,
            fixture.correction.source,
            fixture.correction.timestamp,
        )
    return {"records": by_event, "correction": corrected}


def _citations_support(rubric: QueryRubric, source_ids: list[str] | tuple[str, ...], store: Store | None) -> bool:
    if store is None or not source_ids:
        return False

    covered = set()
    for source_id in source_ids:
        try:
            record = store.get(source_id, as_of=rubric.as_of)
        except (KeyError, ValueError):
            return False
        if not record["is_fact_evidence"]:
            return False
        content = record["content"]
        if rubric.evidence_context_pattern:
            context = re.search(rubric.evidence_context_pattern, content, re.IGNORECASE)
            if context is None:
                return False
            content = context.group()
        supports = {
            label for label, pattern in rubric.expected_patterns.items()
            if re.search(pattern, content, re.IGNORECASE)
        }
        if not supports:
            return False
        covered.update(supports)
    return covered == rubric.expected_patterns.keys()


def score_response(
    rubric: QueryRubric,
    text: str,
    *,
    source_ids: list[str] | tuple[str, ...] = (),
    evidence_store: Store | None = None,
) -> dict:
    """Score an answer and require scoped evidence for citation-bearing rubrics."""
    answer = text or ""
    matched = {
        label: bool(re.search(pattern, answer, re.IGNORECASE))
        for label, pattern in rubric.expected_patterns.items()
    }
    violations = [
        label for label, pattern in rubric.forbidden_patterns.items()
        if re.search(pattern, answer, re.IGNORECASE)
    ]
    citation_ok = not rubric.requires_citations or _citations_support(rubric, source_ids, evidence_store)
    passed = bool(answer.strip()) and all(matched.values()) and not violations and citation_ok
    return {
        "query": rubric.name,
        "route": rubric.route,
        "passed": passed,
        "matched": matched,
        "violations": violations,
        "citation_ok": citation_ok,
        "source_count": len(source_ids),
    }


def verify_fixtures() -> dict:
    """Check history, transient filtering, and cited rubrics in a temporary Store."""
    summary = {"workflows": {}}
    with TemporaryDirectory(prefix="lightreason-sci-fixture-") as tmp:
        for index, (wid, fixture) in enumerate(WORKFLOWS.items()):
            db_path = Path(tmp) / f"fixture-{index}.sqlite3"
            store = Store(db_path, "scientist", wid)
            seeded = seed_workflow_store(store, fixture, apply_correction=True)
            before = store.search(fixture.correction.slot_key, as_of=fixture.initial_as_of, limit=4)
            after = store.search(fixture.correction.slot_key, limit=4)
            transient_hits = store.search("hallway lobby printer", limit=4)
            if transient_hits:
                raise RuntimeError(f"Transient chatter entered durable store for {wid}.")
            if not any(fixture.correction.previous_compact_value.split()[0][:3] in r["content"] for r in before):
                raise RuntimeError(f"Historical as_of query missed original protocol for {wid}.")
            if not any(fixture.correction.compact_value.split()[0][:3] in r["content"] for r in after):
                raise RuntimeError(f"Current query missed corrected protocol for {wid}.")
            for rubric in fixture.queries:
                if not rubric.requires_citations:
                    continue
                original_id = seeded["records"][fixture.correction.target_event_id]["memory_id"]
                protocol = store.get(
                    original_id if rubric.as_of == fixture.initial_as_of else seeded["correction"]["memory_id"],
                    as_of=rubric.as_of,
                )
                score = score_response(
                    rubric, protocol["content"], source_ids=[protocol["memory_id"]], evidence_store=store,
                )
                if not score["passed"]:
                    raise RuntimeError(f"Protocol citation failed rubric {rubric.name} for {wid}.")
            summary["workflows"][wid] = {
                "title": fixture.title,
                "durable_records_seeded": len(seeded["records"]),
                "transient_skipped": sum(obs.is_transient for obs in fixture.observations),
                "correction_memory_id": seeded["correction"]["memory_id"],
                "supersedes_id": seeded["correction"]["supersedes_id"],
                "queries": [q.name for q in fixture.queries],
            }
    summary["unrelated_controls"] = len(UNRELATED_CONTROL_OBSERVATIONS)
    summary["verified"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=tuple(WORKFLOWS), default=ZNO_WORKFLOW.workflow_id)
    parser.add_argument("--verify", action="store_true", help="run store versioning and rubric self-checks")
    parser.add_argument("--seed-db", type=Path, help="seed a private SQLite store with the selected workflow")
    parser.add_argument("--user", default="scientist")
    parser.add_argument("--workspace", default="materials-lab")
    parser.add_argument("--no-correction", action="store_true")
    args = parser.parse_args(argv)

    if args.verify:
        print(json.dumps(verify_fixtures(), indent=2))
        return 0
    fixture = WORKFLOWS[args.workflow]
    if args.seed_db is not None:
        store = Store(args.seed_db, args.user, args.workspace)
        seeded = seed_workflow_store(store, fixture, apply_correction=not args.no_correction)
        print(json.dumps({
            "workflow_id": fixture.workflow_id,
            "seeded_events": sorted(seeded["records"]),
            "corrected": seeded["correction"]["memory_id"] if seeded["correction"] else None,
        }, indent=2))
        return 0
    print(json.dumps({
        "workflow_id": fixture.workflow_id,
        "title": fixture.title,
        "observations": [asdict(obs) for obs in fixture.observations],
        "correction": asdict(fixture.correction),
        "queries": [q.name for q in fixture.queries],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
