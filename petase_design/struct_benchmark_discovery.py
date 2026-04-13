"""
Automated WT/mutant structure-pair discovery for structural benchmarking.

This module queries RCSB for high-quality, single-protein X-ray structures and
builds WT/mutant benchmark pairs grouped by enzyme identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"([A-Z])\s*([0-9]+)\s*([A-Z])")


@dataclass
class StructureCandidate:
    pdb_id: str
    entity_id: str
    chain_id: str
    enzyme_key: str
    enzyme_label: str
    uniprot_id: str | None
    mutation_raw: str | None
    mutation_code: str | None
    is_wildtype: bool
    sequence: str
    resolution_a: float | None
    entry_title: str


@dataclass
class BenchmarkPair:
    enzyme_key: str
    enzyme_label: str
    uniprot_id: str | None
    wt_pdb_id: str
    wt_entity_id: str
    wt_chain_id: str
    wt_sequence: str
    wt_resolution_a: float | None
    mut_pdb_id: str
    mut_entity_id: str
    mut_chain_id: str
    mut_sequence: str
    mut_resolution_a: float | None
    mutation_code: str
    mutation_raw: str | None
    seq_identity: float


@dataclass
class DiscoveryReport:
    pairs: list[BenchmarkPair]
    exclusions: list[dict[str, Any]]
    stats: dict[str, Any]


def _http_json(url: str, *, payload: dict[str, Any] | None = None, timeout_s: int = 45) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url=url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _clean_sequence(seq: str | None) -> str:
    if not seq:
        return ""
    seq = re.sub(r"[^A-Z]", "", seq.upper())
    return "".join(ch for ch in seq if ch in AA_SET)


def _parse_resolution(entry_payload: dict[str, Any]) -> float | None:
    vals = ((entry_payload.get("rcsb_entry_info") or {}).get("resolution_combined") or [])
    if not vals:
        return None
    try:
        return float(vals[0])
    except (TypeError, ValueError):
        return None


def _normalize_mutation(raw_mutation: str | None) -> tuple[bool, str | None]:
    """
    Returns (is_wildtype, mutation_code).

    - WT: mutation annotation absent/unknown
    - Mutant: exactly one missense token like A123V
    - Unsupported mutation forms return (False, None)
    """
    if raw_mutation is None:
        return True, None
    text = str(raw_mutation).strip().upper()
    if text in {"", "?", ".", "NONE", "NOT PROVIDED"}:
        return True, None
    tokens = MUT_RE.findall(text)
    if len(tokens) != 1:
        return False, None
    aa0, pos, aa1 = tokens[0]
    if aa0 not in AA_SET or aa1 not in AA_SET:
        return False, None
    return False, f"{aa0}{int(pos)}{aa1}"


def _sequence_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        m = min(len(a), len(b))
        if m == 0:
            return 0.0
        same = sum(1 for x, y in zip(a[:m], b[:m]) if x == y)
        return same / max(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def _choose_chain_id(polymer_payload: dict[str, Any]) -> str | None:
    ids = (polymer_payload.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
    for chain in ids:
        if chain and isinstance(chain, str):
            return chain
    return None


def _extract_uniprot_id(polymer_payload: dict[str, Any]) -> str | None:
    refs = (polymer_payload.get("rcsb_polymer_entity_container_identifiers") or {}).get("reference_sequence_identifiers") or []
    for item in refs:
        db = str((item or {}).get("database_name", "")).lower()
        acc = str((item or {}).get("database_accession", "")).strip()
        if "uniprot" in db and acc:
            return acc
    return None


def _extract_enzyme_label(polymer_payload: dict[str, Any]) -> str:
    rcsb_poly = polymer_payload.get("rcsb_polymer_entity") or {}
    desc = str(rcsb_poly.get("pdbx_description") or "").strip()
    if desc:
        return desc
    names = rcsb_poly.get("rcsb_ec_lineage") or []
    if names:
        first = names[0] or {}
        label = str(first.get("name") or "").strip()
        if label:
            return label
    ecs = rcsb_poly.get("ec_numbers") or []
    if ecs:
        return f"EC {ecs[0]}"
    return "unknown_enzyme"


def _extract_ec_key(polymer_payload: dict[str, Any]) -> str | None:
    ecs = (polymer_payload.get("rcsb_polymer_entity") or {}).get("ec_numbers") or []
    if not ecs:
        return None
    return ",".join(sorted(str(x) for x in ecs if x))


def _looks_mutant_title(title: str) -> bool:
    t = title.strip().lower()
    if not t:
        return False
    keys = (" mutant", "mutant ", "mutant-", " mutant-", "variant", "substitution")
    return any(k in t for k in keys)


def _derive_single_mutation_code(wt_seq: str, mut_seq: str) -> str | None:
    if len(wt_seq) != len(mut_seq) or not wt_seq:
        return None
    diffs: list[tuple[int, str, str]] = []
    for i, (a, b) in enumerate(zip(wt_seq, mut_seq), start=1):
        if a != b:
            diffs.append((i, a, b))
            if len(diffs) > 1:
                return None
    if len(diffs) != 1:
        return None
    pos, a0, a1 = diffs[0]
    if a0 not in AA_SET or a1 not in AA_SET:
        return None
    return f"{a0}{pos}{a1}"


def _search_entry_ids(*, max_entries: int, resolution_max_a: float) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": float(resolution_max_a),
                    },
                },
            ],
        },
        "request_options": {
            "paginate": {"start": 0, "rows": int(max_entries)},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
        "return_type": "entry",
    }
    payload = _http_json(RCSB_SEARCH_URL, payload=query)
    out: list[str] = []
    for row in payload.get("result_set", []):
        identifier = str((row or {}).get("identifier", "")).strip().upper()
        if identifier:
            out.append(identifier)
    return out


def discover_wt_mutant_pairs(
    *,
    max_entries: int = 800,
    max_enzymes: int = 3,
    max_pairs_per_enzyme: int = 3,
    resolution_max_a: float = 2.8,
    min_seq_identity: float = 0.95,
    max_len_delta_frac: float = 0.05,
    enable_sequence_diff_fallback: bool = True,
) -> DiscoveryReport:
    exclusions: list[dict[str, Any]] = []
    candidates: list[StructureCandidate] = []
    stats: dict[str, Any] = {"searched_entries": 0, "candidate_entries": 0, "fallback_pairs": 0}

    try:
        entry_ids = _search_entry_ids(max_entries=max_entries, resolution_max_a=resolution_max_a)
    except URLError as e:
        raise RuntimeError(f"RCSB search failed: {e}") from e

    stats["searched_entries"] = len(entry_ids)

    for pdb_id in entry_ids:
        try:
            entry_payload = _http_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id.lower()))
            entry_title = str((entry_payload.get("struct") or {}).get("title") or "").strip()
            entity_ids = (entry_payload.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
            if len(entity_ids) != 1:
                exclusions.append({"pdb_id": pdb_id, "reason": "not_single_protein_entity"})
                continue
            entity_id = str(entity_ids[0])
            polymer_payload = _http_json(RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id.lower(), entity_id=entity_id))
            chain_id = _choose_chain_id(polymer_payload)
            if not chain_id:
                exclusions.append({"pdb_id": pdb_id, "reason": "missing_chain_id"})
                continue

            seq = _clean_sequence(((polymer_payload.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can")))
            if len(seq) < 40:
                exclusions.append({"pdb_id": pdb_id, "reason": "sequence_too_short"})
                continue

            mutation_raw = (polymer_payload.get("entity_poly") or {}).get("pdbx_mutation")
            is_wt, mutation_code = _normalize_mutation(mutation_raw)
            if is_wt and mutation_code is None and _looks_mutant_title(entry_title):
                # Many RCSB entries omit machine-readable mutation annotations even for mutants.
                # Promote these to unresolved mutant candidates for sequence-diff fallback.
                is_wt = False
                stats["title_promoted_mutants"] = int(stats.get("title_promoted_mutants", 0)) + 1
            if not is_wt and mutation_code is None:
                exclusions.append(
                    {
                        "pdb_id": pdb_id,
                        "reason": "unsupported_mutation_annotation_kept_for_fallback",
                        "mutation_raw": mutation_raw,
                    }
                )

            enzyme_label = _extract_enzyme_label(polymer_payload)
            uniprot_id = _extract_uniprot_id(polymer_payload)
            ec_key = _extract_ec_key(polymer_payload)
            enzyme_key = uniprot_id or (f"ec:{ec_key}" if ec_key else enzyme_label.lower().replace(" ", "_"))

            candidates.append(
                StructureCandidate(
                    pdb_id=pdb_id,
                    entity_id=entity_id,
                    chain_id=chain_id,
                    enzyme_key=enzyme_key,
                    enzyme_label=enzyme_label,
                    uniprot_id=uniprot_id,
                    mutation_raw=str(mutation_raw) if mutation_raw is not None else None,
                    mutation_code=mutation_code,
                    is_wildtype=is_wt,
                    sequence=seq,
                    resolution_a=_parse_resolution(entry_payload),
                    entry_title=entry_title,
                )
            )
        except Exception as e:  # network/schema drifts should not kill full discovery
            exclusions.append({"pdb_id": pdb_id, "reason": "entry_parse_error", "error": str(e)})

    stats["candidate_entries"] = len(candidates)

    by_enzyme: dict[str, list[StructureCandidate]] = {}
    for c in candidates:
        by_enzyme.setdefault(c.enzyme_key, []).append(c)

    def _pick_best_wt(mut: StructureCandidate, wt_rows: list[StructureCandidate]) -> tuple[StructureCandidate | None, float]:
        best_wt: StructureCandidate | None = None
        best_ident = -1.0
        for wt in wt_rows:
            ident = _sequence_identity(wt.sequence, mut.sequence)
            len_delta = abs(len(wt.sequence) - len(mut.sequence)) / max(len(wt.sequence), len(mut.sequence), 1)
            if len_delta > max_len_delta_frac:
                continue
            if ident < min_seq_identity:
                continue
            if ident > best_ident:
                best_ident = ident
                best_wt = wt
        return best_wt, best_ident

    pairs: list[BenchmarkPair] = []
    used_pair_keys: set[tuple[str, str, str, str]] = set()
    chosen_enzyme_keys = sorted(by_enzyme.keys())
    for enzyme_key in chosen_enzyme_keys:
        if len({p.enzyme_key for p in pairs}) >= int(max_enzymes):
            break

        rows = by_enzyme[enzyme_key]
        wt_rows = [x for x in rows if x.is_wildtype]
        strict_mut_rows = [x for x in rows if not x.is_wildtype and x.mutation_code]
        unresolved_mut_rows = [x for x in rows if not x.is_wildtype and not x.mutation_code]
        if not wt_rows or (not strict_mut_rows and not (enable_sequence_diff_fallback and unresolved_mut_rows)):
            exclusions.append({"enzyme_key": enzyme_key, "reason": "missing_wt_or_mutant_in_group"})
            continue

        wt_rows = sorted(wt_rows, key=lambda r: (r.resolution_a is None, r.resolution_a or 999.0))
        strict_mut_rows = sorted(strict_mut_rows, key=lambda r: (r.resolution_a is None, r.resolution_a or 999.0))
        unresolved_mut_rows = sorted(unresolved_mut_rows, key=lambda r: (r.resolution_a is None, r.resolution_a or 999.0))
        per_enzyme_pairs = 0

        for mut in strict_mut_rows:
            best_wt, best_ident = _pick_best_wt(mut, wt_rows)
            if best_wt is None:
                exclusions.append(
                    {
                        "enzyme_key": enzyme_key,
                        "pdb_id": mut.pdb_id,
                        "reason": "no_comparable_wt_match",
                        "min_seq_identity": min_seq_identity,
                    }
                )
                continue
            pair_key = (enzyme_key, best_wt.pdb_id, mut.pdb_id, str(mut.mutation_code))
            if pair_key in used_pair_keys:
                continue
            used_pair_keys.add(pair_key)

            pairs.append(
                BenchmarkPair(
                    enzyme_key=enzyme_key,
                    enzyme_label=mut.enzyme_label,
                    uniprot_id=mut.uniprot_id,
                    wt_pdb_id=best_wt.pdb_id,
                    wt_entity_id=best_wt.entity_id,
                    wt_chain_id=best_wt.chain_id,
                    wt_sequence=best_wt.sequence,
                    wt_resolution_a=best_wt.resolution_a,
                    mut_pdb_id=mut.pdb_id,
                    mut_entity_id=mut.entity_id,
                    mut_chain_id=mut.chain_id,
                    mut_sequence=mut.sequence,
                    mut_resolution_a=mut.resolution_a,
                    mutation_code=mut.mutation_code or "NA",
                    mutation_raw=mut.mutation_raw,
                    seq_identity=max(best_ident, 0.0),
                )
            )
            per_enzyme_pairs += 1
            if per_enzyme_pairs >= int(max_pairs_per_enzyme):
                break

        if enable_sequence_diff_fallback and per_enzyme_pairs < int(max_pairs_per_enzyme):
            for mut in unresolved_mut_rows:
                if not _looks_mutant_title(mut.entry_title):
                    exclusions.append(
                        {
                            "enzyme_key": enzyme_key,
                            "pdb_id": mut.pdb_id,
                            "reason": "fallback_skipped_title_not_mutant_like",
                            "entry_title": mut.entry_title,
                        }
                    )
                    continue
                best_wt, best_ident = _pick_best_wt(mut, wt_rows)
                if best_wt is None:
                    exclusions.append(
                        {
                            "enzyme_key": enzyme_key,
                            "pdb_id": mut.pdb_id,
                            "reason": "fallback_no_comparable_wt_match",
                            "min_seq_identity": min_seq_identity,
                        }
                    )
                    continue
                derived = _derive_single_mutation_code(best_wt.sequence, mut.sequence)
                if derived is None:
                    exclusions.append(
                        {
                            "enzyme_key": enzyme_key,
                            "pdb_id": mut.pdb_id,
                            "wt_pdb_id": best_wt.pdb_id,
                            "reason": "fallback_not_single_missense_diff",
                        }
                    )
                    continue
                pair_key = (enzyme_key, best_wt.pdb_id, mut.pdb_id, derived)
                if pair_key in used_pair_keys:
                    continue
                used_pair_keys.add(pair_key)
                stats["fallback_pairs"] = int(stats.get("fallback_pairs", 0)) + 1
                pairs.append(
                    BenchmarkPair(
                        enzyme_key=enzyme_key,
                        enzyme_label=mut.enzyme_label,
                        uniprot_id=mut.uniprot_id,
                        wt_pdb_id=best_wt.pdb_id,
                        wt_entity_id=best_wt.entity_id,
                        wt_chain_id=best_wt.chain_id,
                        wt_sequence=best_wt.sequence,
                        wt_resolution_a=best_wt.resolution_a,
                        mut_pdb_id=mut.pdb_id,
                        mut_entity_id=mut.entity_id,
                        mut_chain_id=mut.chain_id,
                        mut_sequence=mut.sequence,
                        mut_resolution_a=mut.resolution_a,
                        mutation_code=derived,
                        mutation_raw=mut.mutation_raw,
                        seq_identity=max(best_ident, 0.0),
                    )
                )
                per_enzyme_pairs += 1
                if per_enzyme_pairs >= int(max_pairs_per_enzyme):
                    break

    stats["selected_pairs"] = len(pairs)
    stats["selected_enzymes"] = len({p.enzyme_key for p in pairs})
    return DiscoveryReport(pairs=pairs, exclusions=exclusions, stats=stats)


def write_discovery_manifest(report: DiscoveryReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2026-04-10.struct-benchmark.discovery.v1",
        "stats": report.stats,
        "pairs": [asdict(p) for p in report.pairs],
        "exclusions": report.exclusions,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
