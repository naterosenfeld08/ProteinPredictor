"""
petase-thermostability-benchmark — browser GUI (Streamlit).

Run from the repo root:
    streamlit run gui/app.py

You can also double-click a small shell script or use Automator; the app opens in your default browser.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

# Repo root on sys.path *before* `from gui.*` (Streamlit loads this file with cwd ≠ repo).
REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_STR = str(REPO_ROOT)
if _REPO_STR not in sys.path:
    sys.path.insert(0, _REPO_STR)

import pandas as pd
import streamlit as st

from gui.insights import render_fireprot_honesty_callout, render_prediction_analytics
from gui.sequence_structure_helper import (
    build_pseudo_pdb_from_sequence,
    fetch_known_structure_pdb,
    find_known_structure_match,
    identify_sequence,
    sanitize_sequence,
)
from gui.structure_view import (
    format_py3dmol_diagnostics,
    render_3dmol_network_help,
    render_structure_background_motion,
    render_structure_panel,
)

DEFAULT_VIZ_STYLE = "cartoon_sticks"
DEFAULT_VIZ_SPIN = False


def _apply_presentation_css() -> None:
    panel_radius = "2px"
    panel_border = "rgba(255,255,255,0.16)"
    panel_blur = "4px"
    panel_bg = "rgba(7,7,8,0.86)"
    header_tracking = "0.04em"
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {{
  --pp-panel-bg: {panel_bg};
  --pp-panel-border: {panel_border};
  --pp-panel-radius: {panel_radius};
  --pp-panel-blur: {panel_blur};
  --pp-text: #f5f6f8;
  --pp-muted: #9ba0aa;
}}

/* Clean dark scientific theme */
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(900px 380px at 10% 0%, rgba(100,120,150,0.08), transparent 55%), #030304;
}}

[data-testid="stAppViewContainer"] > .main {{
  position: relative;
  z-index: 1;
}}
.block-container {{
  padding-top: 1.1rem;
  padding-bottom: 1.3rem;
  max-width: 1400px;
}}
h1, h2, h3 {{
  font-family: "Space Grotesk", "Inter", "Segoe UI", Arial, sans-serif;
  letter-spacing: {header_tracking};
  text-transform: none;
  font-weight: 700;
  color: var(--pp-text);
  line-height: 1.14;
}}
p, li, label, span, div {{
  font-family: "Inter", "SF Pro Display", "Segoe UI", Arial, sans-serif;
  color: var(--pp-text);
}}

/* Panels and cards */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stMetric"],
div[data-testid="stAlert"] {{
  border: 1px solid var(--pp-panel-border) !important;
  border-radius: var(--pp-panel-radius) !important;
  background: var(--pp-panel-bg) !important;
  backdrop-filter: blur(var(--pp-panel-blur));
}}
div[data-testid="stMetric"] {{
  padding: 0.55rem 0.7rem;
}}
button, [data-baseweb="select"] > div, textarea, input {{
  border-radius: 0 !important;
  border-color: rgba(255,255,255,0.22) !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
  border-radius: 0 !important;
  letter-spacing: 0.03em;
  text-transform: none;
  font-size: 0.72rem;
  border: 1px solid transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  border: 1px solid rgba(255,255,255,0.32) !important;
  background: rgba(255,255,255,0.05) !important;
}}
[data-testid="stSidebar"] {{
  background: #060607;
  border-right: 1px solid rgba(255,255,255,0.12);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _poll_subprocess_with_ui(
    proc: subprocess.Popen,
    *,
    title: str,
    detail: str,
    hint_terminal: str = "",
    stages: list[str] | None = None,
    on_tick: Callable[[int, float], None] | None = None,
    max_runtime_seconds: float | None = None,
) -> int:
    """
    Wait for a child process while refreshing Streamlit UI.

    Heavy ML runs in a **separate Python process** (see `predict_worker.py`) so this
    Streamlit process never imports torch — the server can answer WebSocket pings and
    avoids “Is Streamlit still running?” during long embeddings.
    """
    slot = st.empty()
    started = time.monotonic()
    tick = 0
    while proc.poll() is None:
        time.sleep(0.5)
        tick += 1
        elapsed = time.monotonic() - started
        dots = "." * (1 + (tick % 3))
        extra = f"  \n{hint_terminal}" if hint_terminal else ""
        stage_block = ""
        if stages:
            active = (tick // 4) % max(len(stages), 1)
            lines: list[str] = []
            for i, stage in enumerate(stages):
                if i < active:
                    lines.append(f"- [done] {stage}")
                elif i == active:
                    lines.append(f"- [running] {stage}")
                else:
                    lines.append(f"- [pending] {stage}")
            stage_block = "\n\n**Pipeline status**  \n" + "  \n".join(lines)
        slot.info(
            f"**{title}**{dots}  \n"
            f"{detail}  \n\n"
            f"Elapsed **{elapsed:.0f}s**. First run may download large model weights "
            f"(**10–20+ minutes**).  \n\n"
            f"**Leave this tab open** and check the **terminal** where you ran "
            f"`streamlit run` for download / embedding progress.{extra}{stage_block}"
        )
        if on_tick is not None:
            try:
                on_tick(tick, elapsed)
            except Exception:
                # Live visuals are best-effort and should never interrupt a run.
                pass
        if max_runtime_seconds is not None and elapsed >= float(max_runtime_seconds):
            try:
                proc.kill()
            except Exception:
                pass
            slot.error(
                f"{title} exceeded configured runtime limit ({max_runtime_seconds:.0f}s) and was stopped. "
                "Reduce cycles/survivor budget or increase timeout."
            )
            return -9
    slot.empty()
    return int(proc.returncode or 0)


def _build_sequence_visual_payload(seq: str) -> dict[str, str] | None:
    """Resolve best-effort structure payload for a typed sequence."""
    clean = sanitize_sequence(seq)
    if len(clean) < 20:
        return None
    known = find_known_structure_match(clean)
    if known:
        pdb_text, err = fetch_known_structure_pdb(known)
        if pdb_text:
            return {
                "mode": "known_pdb",
                "title": f"{known.get('name', 'Known structure')} ({known.get('pdb_id', '?')}:{known.get('chain', '?')})",
                "detail": known.get("match_detail", "Matched curated known structure."),
                "pdb_text": pdb_text,
            }
        return {
            "mode": "pseudo_fallback",
            "title": "Known structure match (download failed)",
            "detail": f"{known.get('match_detail', '')} PDB fetch failed: {err}",
            "pdb_text": build_pseudo_pdb_from_sequence(clean),
        }
    return {
        "mode": "pseudo",
        "title": "Pseudo structure preview",
        "detail": "No curated known-structure match; showing a pseudo model while prediction runs.",
        "pdb_text": build_pseudo_pdb_from_sequence(clean),
    }


def _discover_recent_structure_paths(root: Path, *, max_items: int = 6) -> list[Path]:
    """Find recent structure files emitted by ColabFold jobs."""
    if not root.exists():
        return []
    out: list[Path] = []
    patterns = ("**/*.pdb", "**/*.cif")
    for pat in patterns:
        out.extend(root.glob(pat))
    files = [p for p in out if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in files:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        dedup.append(p)
        if len(dedup) >= max_items:
            break
    return dedup


def _render_live_colabfold_gallery(container: st.delta_generator.DeltaGenerator, root: Path) -> None:
    """Render the freshest discovered ColabFold structures in a compact gallery."""
    with container.container():
        paths = _discover_recent_structure_paths(root, max_items=3)
        if not paths:
            st.caption("Live ColabFold gallery: waiting for first structure file...")
            return
        st.caption(f"Live ColabFold gallery: {len(paths)} recent structure file(s) discovered.")
        for i, p in enumerate(paths, start=1):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "ATOM" not in text and "HETATM" not in text:
                continue
            with st.expander(f"Model {i}: {p.name}", expanded=(i == 1)):
                render_structure_panel(
                    text,
                    key_prefix=f"live_cf_{i}_{p.stem}",
                    show_controls=False,
                    default_style=DEFAULT_VIZ_STYLE,
                    default_spin=True,
                    height=360,
                    show_troubleshoot_caption=False,
                )


def _default_model_path() -> str:
    candidates = [
        REPO_ROOT
        / "training_output (CRITICAL DIRECTORY DO NOT TOUCH)"
        / "mlp_rf_ensemble_full_both"
        / "mlp_rf_ensemble.pkl",
        REPO_ROOT / "mlp_rf_ensemble.pkl",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return str(candidates[0])


def tab_predict() -> None:
    st.subheader("Single-Sequence Analysis")
    render_fireprot_honesty_callout()
    st.caption(
        "Runs your trained ensemble with PLM embeddings. The first execution may download "
        "ProtT5/ESM model weights and can take several minutes."
    )
    last = st.session_state.get("last_prediction")
    if last:
        with st.expander("Last prediction in this session", expanded=False):
            st.metric("Predicted ΔΔG (kcal/mol)", f"{last['pred_value']:.4f}")
            st.caption(f"Sequence: **{last.get('name', '?')}** ({last.get('length', '?')} aa)")
            if st.button("Clear saved result", key="clear_last_pred"):
                del st.session_state["last_prediction"]
                st.rerun()

    model_path = st.text_input(
        "Path to model pickle",
        value=_default_model_path(),
        help="Usually `mlp_rf_ensemble.pkl` from your training output folder.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        seq_name = st.text_input("Sequence name", value="query1")
    with col_b:
        embed_type = st.selectbox(
            "Embedding mode",
            options=["both", "prot_t5", "esm2"],
            index=0,
            help="Must match how the model was trained (full pipeline uses both + composition).",
        )

    sequence = st.text_area(
        "Amino acid sequence (single letter)",
        height=120,
        placeholder="MKT...",
    )
    use_comp = st.checkbox("Append composition features (20 AA frequencies)", value=True)
    payload = _build_sequence_visual_payload(sequence)
    if payload and payload.get("pdb_text"):
        st.markdown("#### Live Structure Companion")
        st.caption(
            "During embedding and inference, this panel previews the best available structure for "
            "the current sequence (curated known PDB when matched; otherwise pseudo fallback)."
        )
        if payload.get("mode") == "known_pdb":
            st.success(f"{payload.get('title', 'Known structure')} — {payload.get('detail', '')}")
        else:
            st.info(f"{payload.get('title', 'Preview')} — {payload.get('detail', '')}")
        render_structure_background_motion(payload["pdb_text"], key_prefix="predict_live_bg")
        with st.expander("Interactive structure preview", expanded=False):
            render_structure_panel(
                payload["pdb_text"],
                key_prefix="predict_live_full",
                default_style=DEFAULT_VIZ_STYLE,
                default_spin=True,
                height=460,
            )

    if st.button("Run prediction", type="primary"):
        if not model_path or not Path(model_path).is_file():
            st.error("Model file not found. Train a model or fix the path.")
            return
        seq = "".join(sequence.split()).upper()
        if len(seq) < 10:
            st.error("Sequence looks too short (min length 10).")
            return

        fasta_content = f">{seq_name}\n{seq}\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(fasta_content)
            tmp_path = tmp.name

        worker = REPO_ROOT / "gui" / "predict_worker.py"
        if not worker.is_file():
            st.error(f"Missing worker script: {worker}")
            Path(tmp_path).unlink(missing_ok=True)
            return

        out_fd, out_path = tempfile.mkstemp(suffix=".json", text=True)
        os.close(out_fd)
        out_path_p = Path(out_path)

        cmd = [
            sys.executable,
            str(worker),
            "--fasta",
            tmp_path,
            "--model",
            model_path,
            "--embedding-model-type",
            embed_type,
            "--out",
            str(out_path_p),
        ]
        if not use_comp:
            cmd.append("--no-composition")

        try:
            # Child loads torch/transformers; parent only polls (keeps WS alive).
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            code = _poll_subprocess_with_ui(
                proc,
                title="Embedding + predicting (separate process)",
                detail="A second Python process is computing embeddings and running the model.",
                hint_terminal="You should see ProtT5/ESM progress **in the terminal**, not only in the browser.",
            )
            raw = ""
            try:
                raw = out_path_p.read_text(encoding="utf-8").strip()
            except OSError:
                pass

            if code < 0:
                sig = -code
                st.error(
                    f"Prediction worker **crashed** (Unix signal **{sig}**, often **SIGSEGV** on macOS). "
                    "This usually comes from PyTorch / tokenizers / OpenMP threading when embedding. "
                    "The worker now forces safer env vars (`TOKENIZERS_PARALLELISM=false`, single BLAS thread, "
                    "`KMP_DUPLICATE_LIB_OK`); **restart Streamlit and try again**. "
                    "If it persists, run the same prediction from a plain terminal: "
                    "`python predict.py fasta your.fasta --model_path your.pkl`"
                )
                if raw:
                    st.code(raw[:4000])
                else:
                    st.caption("No JSON was written — the process died during native code (e.g. first ProtT5 forward pass).")
                return

            if not raw:
                st.error(
                    f"Worker produced **empty output** (exit code {code}). "
                    "See the terminal where `streamlit run` is running for errors."
                )
                return

            try:
                results = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid worker JSON ({code=}): {e}")
                st.code(raw[:2000])
                return

            if not isinstance(results, dict):
                st.error(
                    "Worker returned JSON that is not an object (expected a dict with "
                    "`predictions` or `ok`/`error`). Got: "
                    f"{type(results).__name__}"
                )
                st.code(raw[:2000])
                return

            if code != 0 or results.get("ok") is False:
                err = results.get("error", raw)
                st.error("Prediction worker failed.")
                st.code(str(err)[:8000])
                return
        except Exception as e:
            st.exception(e)
            return
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            out_path_p.unlink(missing_ok=True)

        preds = results.get("predictions") or []
        if not preds:
            st.warning("No predictions returned. Check validation / model compatibility.")
            st.json(results)
            return

        p0 = preds[0]
        st.success("Done.")
        st.metric("Predicted ΔΔG (kcal/mol)", f"{p0['pred_value']:.4f}")
        if p0.get("uncertainty") is not None:
            u = float(p0["uncertainty"])
            lo = p0["pred_value"] - 1.96 * u
            hi = p0["pred_value"] + 1.96 * u
            st.caption(f"Approx. 95% interval (RF tree spread): [{lo:.4f}, {hi:.4f}]")
        render_prediction_analytics(p0, seq_name=seq_name, seq_len=len(seq))
        with st.expander("Raw prediction JSON", expanded=False):
            st.json(p0)
        st.session_state["last_prediction"] = {
            "pred_value": float(p0["pred_value"]),
            "name": seq_name,
            "length": len(seq),
            "raw": p0,
        }

    st.markdown("---")
    _render_structure_workspace(key_prefix="predict")


def _petase_results_dataframe(rows: list) -> pd.DataFrame:
    """Column order: key structure / SASA / composite fields first, then the rest."""
    df = pd.json_normalize(rows)
    priority = [
        "generation",
        "generator_policy",
        "parent_ids",
        "archive_member",
        "selected_by",
        "rescue_reason",
        "pareto_rank",
        "objective_scalar",
        "objective_terms.ddg_effective",
        "objective_terms.physics_composite_rank",
        "objective_terms.structure_confidence",
        "objective_terms.novelty_score",
        "objective_terms.catalytic_safety_score",
        "objective_terms.catalytic_safety_penalty",
        "hybrid_score",
        "cheap_score_norm",
        "ddg_pred",
        "ddg_uncertainty",
        "ddg_effective",
        "selected_for_structure",
        "structure_pdb",
        "physics.composite",
        "physics.sasa_total_area",
        "physics.apolar_sasa_fraction",
        "physics.radius_of_gyration",
        "physics.structure_confidence",
        "physics.structural_viability_penalty",
        "physics.openmm_total_energy_kj_mol",
        "physics.openmm_energy_per_residue_kj_mol",
        "physics.mutation_count",
        "physics.active_site_violation",
        "physics.mean_hydrophobicity",
        "physics.net_charge_proxy",
    ]
    head = [c for c in priority if c in df.columns]
    tail = [c for c in df.columns if c not in head]
    out = df[head + tail]
    if "pareto_rank" in out.columns:
        out["pareto_rank"] = pd.to_numeric(out["pareto_rank"], errors="coerce").fillna(9999).astype(int)
    if "objective_scalar" in out.columns:
        out["objective_scalar"] = pd.to_numeric(out["objective_scalar"], errors="coerce").fillna(0.0)
    return out


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_index_tokens(raw: str) -> list[int]:
    """
    Parse tokens like: 10,12,20-25 into sorted unique indices.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    out: set[int] = set()
    for tok in text.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a_s, b_s = tok.split("-", 1)
            a = int(a_s.strip())
            b = int(b_s.strip())
            if b < a:
                a, b = b, a
            for i in range(a, b + 1):
                out.add(int(i))
        else:
            out.add(int(tok))
    return sorted(out)


def _parse_region_budgets(raw: str) -> list[tuple[int, int, int]]:
    """
    Parse lines in format: start-end:max_mut
    Example:
      0-35:1
      36-100:2
    """
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if ":" not in ln or "-" not in ln:
            raise ValueError(f"Invalid region budget line: {ln!r}")
        left, right = ln.split(":", 1)
        a_s, b_s = left.split("-", 1)
        a = int(a_s.strip())
        b = int(b_s.strip())
        m = int(right.strip())
        if b < a:
            a, b = b, a
        if m < 0:
            raise ValueError(f"max_mut must be >= 0: {ln!r}")
        out.append((a, b, m))
    return out


def _profile_defaults(profile: str) -> dict[str, float]:
    presets: dict[str, dict[str, float]] = {
        "explore": {
            "policy_random_frac": 0.55,
            "policy_adaptive_frac": 0.25,
            "policy_recombine_frac": 0.20,
            "obj_w_ddg": 0.25,
            "obj_w_phys": 0.20,
            "obj_w_struct": 0.15,
            "obj_w_nov": 0.30,
            "obj_w_safe": 0.10,
        },
        "balanced": {
            "policy_random_frac": 0.50,
            "policy_adaptive_frac": 0.35,
            "policy_recombine_frac": 0.15,
            "obj_w_ddg": 0.35,
            "obj_w_phys": 0.25,
            "obj_w_struct": 0.15,
            "obj_w_nov": 0.15,
            "obj_w_safe": 0.10,
        },
        "exploit": {
            "policy_random_frac": 0.30,
            "policy_adaptive_frac": 0.45,
            "policy_recombine_frac": 0.25,
            "obj_w_ddg": 0.45,
            "obj_w_phys": 0.25,
            "obj_w_struct": 0.15,
            "obj_w_nov": 0.05,
            "obj_w_safe": 0.10,
        },
    }
    return presets.get(profile, presets["balanced"])


def _validate_petase_config(
    *,
    use_cf: bool,
    cf_bin: str,
    policy_random_frac: float,
    policy_adaptive_frac: float,
    policy_recombine_frac: float,
    obj_w_ddg: float,
    obj_w_phys: float,
    obj_w_struct: float,
    obj_w_nov: float,
    obj_w_safe: float,
    hybrid_cheap_weight: float,
    hybrid_ddg_weight: float,
) -> list[str]:
    """Return blocking validation errors for PETase run config."""
    errors: list[str] = []
    if use_cf:
        cf_bin_path = Path(str(cf_bin)).expanduser()
        if not cf_bin_path.is_file() and shutil.which(str(cf_bin)) is None:
            errors.append(
                f"ColabFold executable not found: `{cf_bin}`. "
                "Provide a valid command on PATH or an absolute executable path."
            )
    if float(policy_random_frac + policy_adaptive_frac + policy_recombine_frac) <= 0:
        errors.append("Policy mix must have a positive sum.")
    if float(obj_w_ddg + obj_w_phys + obj_w_struct + obj_w_nov + obj_w_safe) <= 0:
        errors.append("Objective weights must have a positive sum.")
    if float(hybrid_cheap_weight + hybrid_ddg_weight) <= 0:
        errors.append("Hybrid cheap/ddG weights must have a positive sum.")
    return errors


def _load_run_summary_for_jsonl(out_jsonl: Path) -> dict | None:
    summary_path = out_jsonl.parent / "run_summary.json"
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _render_run_report_cards(summary: dict) -> None:
    counts = summary.get("counts") or {}
    runtime = summary.get("runtime") or {}
    run_meta = summary.get("run") or {}
    n_variants = int(counts.get("n_variants", 0) or 0)
    n_struct = int(counts.get("n_with_structure", 0) or 0)
    n_sasa = int(counts.get("n_with_sasa", 0) or 0)
    n_pareto = int(counts.get("pareto_frontier_count", 0) or 0)
    struct_pct = (100.0 * n_struct / n_variants) if n_variants else 0.0
    sasa_pct = (100.0 * n_sasa / n_variants) if n_variants else 0.0
    pareto_pct = (100.0 * n_pareto / n_variants) if n_variants else 0.0
    runtime_s = _safe_float(runtime.get("seconds_wall")) or 0.0

    st.markdown("#### Run Summary Cards")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Variants", n_variants)
    c2.metric("With structure", f"{n_struct} ({struct_pct:.1f}%)")
    c3.metric("With SASA", f"{n_sasa} ({sasa_pct:.1f}%)")
    c4.metric("Runtime (s)", f"{runtime_s:.3f}")
    c5.metric("Top-K mode", str(run_meta.get("structure_top_k") or "off"))
    c6.metric("Pareto frontier", f"{n_pareto} ({pareto_pct:.1f}%)")

    comp = summary.get("composition") or {}
    if comp:
        pcounts = comp.get("generator_policy_counts") or {}
        scounts = comp.get("selected_by_counts") or {}
        if pcounts:
            st.caption(f"Generator policy mix: {pcounts}")
        if scounts:
            st.caption(f"Selection lane mix: {scounts}")

    top = summary.get("top_variants") or []
    if top:
        st.caption("Top 10 variants from the run summary")
        st.dataframe(pd.DataFrame(top), width="stretch", height=280)


def _render_phase2_analytics(rows: list[dict], *, key_prefix: str) -> None:
    if not rows:
        return
    df = _petase_results_dataframe(rows)
    if "generation" in df.columns:
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

    st.markdown("#### Statistical Analytics")
    a1, a2, a3 = st.columns(3)
    best = None
    if "physics.composite" in df.columns:
        comp = pd.to_numeric(df["physics.composite"], errors="coerce")
        if comp.notna().any():
            best = float(comp.max())
            a1.metric("Best composite", f"{best:.4f}")
            a2.metric("Mean composite", f"{float(comp.mean()):.4f}")
            a3.metric("Composite std", f"{float(comp.std(ddof=0)):.4f}")

    if "generation" in df.columns and "physics.composite" in df.columns:
        trend = (
            df[["generation", "physics.composite"]]
            .dropna()
            .groupby("generation", as_index=False)
            .agg(composite_mean=("physics.composite", "mean"), composite_best=("physics.composite", "max"))
            .sort_values("generation")
        )
        if not trend.empty:
            st.caption("Composite trend by generation")
            st.line_chart(
                trend.set_index("generation")[["composite_mean", "composite_best"]],
                height=230,
                width="stretch",
            )

    s1, s2 = st.columns(2)
    with s1:
        if {"generation", "physics.composite"}.issubset(df.columns):
            scatter = df[["generation", "physics.composite"]].dropna()
            if not scatter.empty:
                st.caption("Generation vs composite")
                st.scatter_chart(
                    scatter.rename(columns={"generation": "x", "physics.composite": "y"}),
                    x="x",
                    y="y",
                    height=260,
                    width="stretch",
                )
    with s2:
        if {"physics.sasa_total_area", "physics.composite"}.issubset(df.columns):
            s_scatter = df[["physics.sasa_total_area", "physics.composite"]].dropna()
            if not s_scatter.empty:
                st.caption("SASA vs composite")
                st.scatter_chart(
                    s_scatter.rename(columns={"physics.sasa_total_area": "x", "physics.composite": "y"}),
                    x="x",
                    y="y",
                    height=260,
                    width="stretch",
                )

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dist_candidates = [
        c
        for c in numeric_cols
        if c.startswith("physics.") or c.startswith("objective_") or c.startswith("objective_terms.")
    ]
    if dist_candidates:
        pick = st.selectbox(
            "Distribution metric",
            options=dist_candidates,
            key=f"{key_prefix}_dist_metric",
        )
        series = pd.to_numeric(df[pick], errors="coerce").dropna()
        if not series.empty:
            st.caption(f"Distribution of `{pick}`")
            bins = pd.cut(series, bins=20)
            counts = bins.value_counts(sort=False)
            labels = [
                f"{iv.left:.4f}-{iv.right:.4f}" if hasattr(iv, "left") else str(iv)
                for iv in counts.index
            ]
            hist_df = pd.DataFrame({"bin": labels, "count": counts.to_numpy()}).set_index("bin")
            st.bar_chart(hist_df, height=220, width="stretch")

    corr_cols = [
        c
        for c in numeric_cols
        if c.startswith("physics.") or c.startswith("objective_") or c.startswith("objective_terms.")
    ]
    if len(corr_cols) >= 2:
        st.caption("Correlation matrix (physics + objective features)")
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        st.dataframe(corr, width="stretch", height=260)


def _render_last_petase_summary() -> None:
    run = st.session_state.get("last_petase_run")
    if not run:
        return
    with st.expander("Last PETase run (this session)", expanded=False):
        st.markdown(
            f"- **JSONL:** `{run.get('out_path', '')}`  \n"
            f"- **Structures dir:** `{run.get('work_root', '')}`  \n"
            f"- **Variants written:** {run.get('n_rows', '?')} (cycles requested: {run.get('n_cycles', '?')}, "
            f"mutations/variant: {run.get('n_mut', '?')}, seed {run.get('seed', '?')})  \n"
            f"- **ColabFold:** {'yes' if run.get('use_colabfold') else 'no'}  \n"
            f"- **Two-stage top-K:** {run.get('top_k') if run.get('use_topk') else 'off'}  \n"
            f"- **Selected for structure:** {run.get('selected_for_structure', '?')}  \n"
            f"- **Rows with `structure_pdb`:** {run.get('with_structure', '?')}  \n"
            f"- **Rows with SASA (need PDB + freesasa):** {run.get('with_sasa', '?')}  \n"
            f"- **Rows scored by ddG stage:** {run.get('with_ddg', '?')} "
            f"(budget {100.0 * float(run.get('ddg_survivor_pct', 0.0)):.0f}%)  \n"
            f"- **Rescued by lane C:** {run.get('rescued', '?')}  \n"
        )
        if run.get("use_colabfold") and run.get("with_structure", 0) == 0:
            st.warning(
                "No structures linked — check each job folder under the structures dir for "
                "**`colabfold.stderr.log`** (full command, stderr/stdout, and a **structure discovery** "
                "file list). ColabFold naming varies by version; the log helps align outputs."
            )


def _render_pipeline_storyboard(*, use_cf: bool, use_topk: bool) -> None:
    stages = [
        "Generate variants",
        "Embed + LLM score context" if use_cf else "Sequence-first scoring pass",
        "Physics composite ranking",
    ]
    if use_cf and use_topk:
        stages.append("Select top-K for ColabFold")
    if use_cf:
        stages.append("Run ColabFold structures")
        stages.append("Re-score with structure/SASA")
    stages.append("Finalize run summary artifact")
    st.markdown("#### Pipeline Overview")
    st.caption("This workflow combines sequence-level priors with optional structure-aware rescoring.")
    cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        cols[i].markdown(f"**{i+1}. {stage}**")


def _render_variant_detail_drawer(rows: list[dict]) -> None:
    if not rows:
        return
    st.markdown("#### Variant detail drawer")
    labels = [str(r.get("job_id", f"row_{i}")) for i, r in enumerate(rows)]
    selected = st.selectbox("Inspect variant", options=labels, index=0, key="variant_drawer_pick")
    row = next((r for r in rows if str(r.get("job_id")) == selected), rows[0])
    phys = row.get("physics") or {}
    muts = row.get("mutations") or []
    muts_txt = ", ".join(f"{m.get('index')}->{m.get('to')}" for m in muts if isinstance(m, dict)) or "none"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite", f"{float(phys.get('composite', 0.0)):.4f}")
    c2.metric("Mutation count", int(phys.get("mutation_count", 0) or 0))
    c3.metric("Selected for structure", "yes" if row.get("selected_for_structure") else "no")
    c4.metric("Has structure", "yes" if row.get("structure_pdb") else "no")
    c5, c6, c7 = st.columns(3)
    c5.metric("Pareto rank", int(row.get("pareto_rank", 0) or 0))
    c6.metric("Objective scalar", f"{float(row.get('objective_scalar', 0.0)):.4f}")
    c7.metric("Generator policy", str(row.get("generator_policy", "n/a")))
    st.caption(f"Mutations: {muts_txt}")
    terms = row.get("objective_terms") or {}
    if isinstance(terms, dict) and terms:
        st.markdown("**Promotion / demotion explanation**")
        tcols = st.columns(5)
        tcols[0].metric("ddG effective", f"{float(terms.get('ddg_effective', 0.0)):.3f}")
        tcols[1].metric("Physics rank", f"{float(terms.get('physics_composite_rank', 0.0)):.3f}")
        tcols[2].metric("Structure conf", f"{float(terms.get('structure_confidence', 0.0)):.3f}")
        tcols[3].metric("Novelty", f"{float(terms.get('novelty_score', 0.0)):.3f}")
        tcols[4].metric("Catalytic safety", f"{float(terms.get('catalytic_safety_score', 0.0)):.3f}")
        notes: list[str] = []
        if float(terms.get("catalytic_safety_penalty", 0.0)) > 1.0:
            notes.append("High catalytic safety penalty depressed objective score.")
        if float(terms.get("novelty_score", 0.0)) < 0.1:
            notes.append("Low novelty indicates conservative design neighborhood.")
        if float(row.get("structure_viable", 1.0)) == 0.0:
            notes.append("Failed structure viability gate and was excluded from ddG stage.")
        if not notes:
            notes.append("No major penalties detected; rank mainly driven by objective weights.")
        st.caption(" ".join(notes))

    structure_pdb = row.get("structure_pdb")
    if structure_pdb:
        p = Path(str(structure_pdb))
        if p.is_file():
            pdb_text = p.read_text(encoding="utf-8", errors="replace")
            render_structure_panel(
                pdb_text,
                key_prefix=f"drawer_{selected}",
                default_style=DEFAULT_VIZ_STYLE,
                default_spin=DEFAULT_VIZ_SPIN,
                height=500,
            )
        else:
            st.warning(f"Structure path not found on disk: `{p}`")


def tab_petase() -> None:
    st.subheader("PETase Batch Design")
    st.caption("Configure variant generation, optional ColabFold structure passes, and hybrid reranking analytics.")
    _render_last_petase_summary()

    col_left, col_right = st.columns([1.1, 1.3], gap="large")
    with col_left:
        st.markdown("#### Run configuration")
        wt = st.text_input(
            "WT FASTA path",
            value=str(REPO_ROOT / "petase_design" / "data" / "petase_6eqd_chainA_notag.fasta"),
        )
        n_cycles = st.number_input("Cycles (variants)", min_value=1, max_value=10_000, value=20)
        n_mut = st.number_input("Mutations per variant", min_value=1, max_value=50, value=2)
        seed = st.number_input("Random seed", value=42)
        out_path = st.text_input(
            "Output JSONL",
            value=str(REPO_ROOT / "petase_design_runs" / "gui_run.jsonl"),
        )
        st.markdown("#### Hybrid reranking (cheap + ddG)")
        ddg_model_path = st.text_input(
            "ddG model (.pkl)",
            value=_default_model_path(),
            help="Used in stage-2 reranking. Lower predicted ddG is preferred.",
        )
        ddg_survivor_pct = st.slider(
            "Stage-2 ddG survivor budget (% of total variants)",
            min_value=5,
            max_value=100,
            value=35,
            step=5,
        )
        c_hw1, c_hw2 = st.columns(2)
        with c_hw1:
            hybrid_cheap_weight = st.number_input("Cheap score weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        with c_hw2:
            hybrid_ddg_weight = st.number_input("ddG prior weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        ddg_uncertainty_lambda = st.number_input(
            "ddG uncertainty penalty (lambda)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.05,
        )
        ddg_embed_type = st.selectbox(
            "ddG embedding mode",
            options=["both", "prot_t5", "esm2"],
            index=0,
        )
        ddg_no_comp = st.checkbox(
            "Disable composition features for ddG stage",
            value=False,
            help="Only enable if your ddG model was trained without composition features.",
        )
        profile = st.selectbox(
            "Search profile",
            options=["balanced", "explore", "exploit", "custom"],
            index=0,
            help="Prefills policy/objective defaults; custom leaves values as-is.",
        )
        if profile != "custom":
            defaults = _profile_defaults(profile)
            for k, v in defaults.items():
                st.session_state[f"petase_{k}"] = float(v)
        with st.expander("Advanced generation + objective controls", expanded=False):
            st.caption("Tune policy mix, archive behavior, objective weighting, and OpenMM stage.")
            p1, p2, p3 = st.columns(3)
            with p1:
                policy_random_frac = st.number_input(
                    "Policy random",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_policy_random_frac", 0.50)),
                    step=0.05,
                    key="petase_policy_random_frac",
                )
            with p2:
                policy_adaptive_frac = st.number_input(
                    "Policy adaptive",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_policy_adaptive_frac", 0.35)),
                    step=0.05,
                    key="petase_policy_adaptive_frac",
                )
            with p3:
                policy_recombine_frac = st.number_input(
                    "Policy recombine",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_policy_recombine_frac", 0.15)),
                    step=0.05,
                    key="petase_policy_recombine_frac",
                )
            archive_size = st.number_input("Pareto archive size", min_value=4, max_value=5000, value=24, step=4)
            no_pareto_archive = st.checkbox("Disable Pareto archive guidance", value=False)
            use_openmm_stage = st.checkbox("Enable OpenMM stage (slow)", value=False)
            openmm_platform = st.text_input("OpenMM platform", value="CPU")
            ddg_max_survivors = st.number_input(
                "ddG max survivors (runtime guard)",
                min_value=1,
                max_value=2000,
                value=64,
                step=1,
                help="Caps how many variants are sent to ddG embeddings regardless of survivor %.",
            )
            max_run_minutes = st.number_input(
                "Max PETase worker runtime (minutes)",
                min_value=1,
                max_value=600,
                value=45,
                step=1,
                help="Safety watchdog; worker is stopped if this limit is exceeded.",
            )
            st.markdown("Objective scalar weights")
            o1, o2, o3, o4, o5 = st.columns(5)
            with o1:
                obj_w_ddg = st.number_input(
                    "w_ddg",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_obj_w_ddg", 0.35)),
                    step=0.05,
                    key="petase_obj_w_ddg",
                )
            with o2:
                obj_w_phys = st.number_input(
                    "w_phys",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_obj_w_phys", 0.25)),
                    step=0.05,
                    key="petase_obj_w_phys",
                )
            with o3:
                obj_w_struct = st.number_input(
                    "w_struct",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_obj_w_struct", 0.15)),
                    step=0.05,
                    key="petase_obj_w_struct",
                )
            with o4:
                obj_w_nov = st.number_input(
                    "w_novelty",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_obj_w_nov", 0.15)),
                    step=0.05,
                    key="petase_obj_w_nov",
                )
            with o5:
                obj_w_safe = st.number_input(
                    "w_safe",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(st.session_state.get("petase_obj_w_safe", 0.10)),
                    step=0.05,
                    key="petase_obj_w_safe",
                )
            pol_sum = float(policy_random_frac + policy_adaptive_frac + policy_recombine_frac)
            obj_sum = float(obj_w_ddg + obj_w_phys + obj_w_struct + obj_w_nov + obj_w_safe)
            if pol_sum > 0:
                st.caption(
                    f"Policy mix (normalized): random {policy_random_frac/pol_sum:.2f}, "
                    f"adaptive {policy_adaptive_frac/pol_sum:.2f}, recombine {policy_recombine_frac/pol_sum:.2f}"
                )
            else:
                st.warning("Policy mix sum is 0; run is invalid until at least one policy > 0.")
            if obj_sum > 0:
                st.caption(
                    f"Objective mix (normalized): ddG {obj_w_ddg/obj_sum:.2f}, physics {obj_w_phys/obj_sum:.2f}, "
                    f"struct {obj_w_struct/obj_sum:.2f}, novelty {obj_w_nov/obj_sum:.2f}, safety {obj_w_safe/obj_sum:.2f}"
                )
            else:
                st.warning("Objective weights sum is 0; run is invalid until at least one objective > 0.")
            st.markdown("Constraint editor")
            protected_idx_text = st.text_area(
                "Protected residue indices (0-based; comma or ranges, e.g. 10,12,40-45)",
                value="",
                height=70,
            )
            region_budget_text = st.text_area(
                "Region mutation budgets (one per line: start-end:max_mut)",
                value="",
                height=90,
            )

        use_cf = st.checkbox("Run ColabFold for each variant (very slow)", value=False)
        use_topk = st.checkbox(
            "Efficiency mode: cheap-score all, run ColabFold only on top variants",
            value=False,
            help="Two-stage screening: sequence-only composite for all, then structure only on top-K.",
        )
        top_k = st.number_input(
            "Top-K variants for ColabFold (when efficiency mode on)",
            min_value=1,
            max_value=10_000,
            value=5,
            disabled=not use_topk,
        )
        cf_bin = st.text_input("colabfold_batch command", value="colabfold_batch")
        num_recycle = st.number_input("ColabFold num-recycle", min_value=0, max_value=12, value=3)
        use_amber = st.checkbox("ColabFold --amber (OpenMM relax)", value=False)
        cf_overwrite = st.checkbox(
            "ColabFold overwrite existing results",
            value=False,
            disabled=not use_cf,
            help="Passes --overwrite-existing-results to colabfold_batch (recompute even if cached).",
        )
        run_clicked = st.button("Run design loop", type="primary")
    with col_right:
        _render_pipeline_storyboard(use_cf=bool(use_cf), use_topk=bool(use_topk and use_cf))
        st.caption("ColabFold progress streams in the terminal while this panel tracks workflow status.")

    if run_clicked:
        wt_p = Path(wt)
        if not wt_p.is_file():
            st.error(f"WT FASTA not found: {wt_p}")
            return
        ddg_model_p = Path(ddg_model_path).expanduser()
        if not ddg_model_p.is_file():
            st.error(f"ddG model not found: {ddg_model_p}")
            return
        config_errors = _validate_petase_config(
            use_cf=bool(use_cf),
            cf_bin=str(cf_bin),
            policy_random_frac=float(policy_random_frac),
            policy_adaptive_frac=float(policy_adaptive_frac),
            policy_recombine_frac=float(policy_recombine_frac),
            obj_w_ddg=float(obj_w_ddg),
            obj_w_phys=float(obj_w_phys),
            obj_w_struct=float(obj_w_struct),
            obj_w_nov=float(obj_w_nov),
            obj_w_safe=float(obj_w_safe),
            hybrid_cheap_weight=float(hybrid_cheap_weight),
            hybrid_ddg_weight=float(hybrid_ddg_weight),
        )
        if config_errors:
            for msg in config_errors:
                st.error(msg)
            return
        if use_openmm_stage and not use_cf:
            st.warning("OpenMM stage requires structures; enable ColabFold to populate structure PDBs.")
        est_survivors = max(1, int(round(float(n_cycles) * (float(ddg_survivor_pct) / 100.0))))
        if est_survivors > int(ddg_max_survivors):
            st.warning(
                f"Estimated ddG survivors ({est_survivors}) exceed cap ({int(ddg_max_survivors)}); "
                "the worker will truncate survivors to the cap."
            )
        try:
            protected_indices = _parse_index_tokens(protected_idx_text)
            region_budgets = _parse_region_budgets(region_budget_text)
        except ValueError as e:
            st.error(f"Constraint parsing failed: {e}")
            return
        if protected_indices and any(i < 0 for i in protected_indices):
            st.error("Protected indices must be >= 0.")
            return
        if region_budgets:
            for s, e, m in region_budgets:
                if s < 0 or e < 0:
                    st.error("Region budget indices must be >= 0.")
                    return
                if m < 0:
                    st.error("Region budget max_mut must be >= 0.")
                    return

        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        worker = REPO_ROOT / "gui" / "design_worker.py"
        if not worker.is_file():
            st.error(f"Missing worker script: {worker}")
            return

        res_fd, res_path = tempfile.mkstemp(suffix=".json", text=True)
        os.close(res_fd)
        result_path = Path(res_path)
        result_path.unlink(missing_ok=True)

        work_root = REPO_ROOT / "petase_design_runs" / "structures"

        cmd = [
            sys.executable,
            str(worker),
            "--wt-fasta",
            str(wt_p.resolve()),
            "--cycles",
            str(int(n_cycles)),
            "--mutations-per-variant",
            str(int(n_mut)),
            "--seed",
            str(int(seed)),
            "--out-jsonl",
            str(out_p.resolve()),
            "--work-root",
            str(work_root.resolve()),
            "--result-json",
            str(result_path.resolve()),
        ]
        if use_cf:
            cmd.append("--colabfold")
            cmd.extend(["--colabfold-bin", cf_bin])
            cmd.extend(["--num-recycle", str(int(num_recycle))])
            if use_topk:
                cmd.extend(["--structure-top-k", str(int(top_k))])
            if use_amber:
                cmd.append("--amber")
            if cf_overwrite:
                cmd.append("--colabfold-overwrite")
        cmd.extend(["--ddg-model", str(ddg_model_p)])
        cmd.extend(["--ddg-survivor-pct", str(float(ddg_survivor_pct) / 100.0)])
        cmd.extend(["--ddg-max-survivors", str(int(ddg_max_survivors))])
        cmd.extend(["--ddg-embedding-model-type", str(ddg_embed_type)])
        cmd.extend(["--hybrid-cheap-weight", str(float(hybrid_cheap_weight))])
        cmd.extend(["--hybrid-ddg-weight", str(float(hybrid_ddg_weight))])
        cmd.extend(["--ddg-uncertainty-lambda", str(float(ddg_uncertainty_lambda))])
        cmd.extend(["--policy-random-frac", str(float(policy_random_frac))])
        cmd.extend(["--policy-adaptive-frac", str(float(policy_adaptive_frac))])
        cmd.extend(["--policy-recombine-frac", str(float(policy_recombine_frac))])
        cmd.extend(["--archive-size", str(int(archive_size))])
        if no_pareto_archive:
            cmd.append("--no-pareto-archive")
        if use_openmm_stage:
            cmd.append("--openmm-stage")
            cmd.extend(["--openmm-platform", str(openmm_platform)])
        cmd.extend(["--objective-ddg-weight", str(float(obj_w_ddg))])
        cmd.extend(["--objective-physics-weight", str(float(obj_w_phys))])
        cmd.extend(["--objective-structure-weight", str(float(obj_w_struct))])
        cmd.extend(["--objective-novelty-weight", str(float(obj_w_nov))])
        cmd.extend(["--objective-catalytic-safety-weight", str(float(obj_w_safe))])
        if protected_indices:
            cmd.extend(["--protected-indices-json", json.dumps(protected_indices)])
        if region_budgets:
            cmd.extend(["--region-budgets-json", json.dumps([[int(s), int(e), int(m)] for s, e, m in region_budgets])])
        if ddg_no_comp:
            cmd.append("--ddg-no-composition")

        try:
            live_gallery_slot = st.empty()
            live_log_slot = st.empty()
            worker_log = REPO_ROOT / "petase_design_runs" / "gui_design_worker.log"
            worker_log.parent.mkdir(parents=True, exist_ok=True)

            def _on_tick_live_gallery(tick: int, _elapsed: float) -> None:
                if use_cf and tick % 8 == 0:
                    _render_live_colabfold_gallery(live_gallery_slot, work_root)
                try:
                    txt = worker_log.read_text(encoding="utf-8", errors="replace")
                    lines = txt.splitlines()
                    if lines:
                        tail = "\n".join(lines[-25:])
                        with live_log_slot.container():
                            st.caption("Worker live log (tail)")
                            st.code(tail)
                except OSError:
                    pass

            with worker_log.open("w", encoding="utf-8") as logf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            code = _poll_subprocess_with_ui(
                proc,
                title=f"Design loop ({n_cycles} cycles, separate process)",
                detail="Proposing variants and scoring. ColabFold makes this much slower.",
                stages=[
                    "Generate variants",
                    "Embedding + scoring",
                    "Composite ranking",
                    "ColabFold jobs" if use_cf else "Skip structures",
                    "Finalize outputs",
                ],
                on_tick=_on_tick_live_gallery,
                max_runtime_seconds=float(max_run_minutes) * 60.0,
            )
            raw = ""
            try:
                raw = result_path.read_text(encoding="utf-8").strip()
            except OSError:
                pass

            if code < 0:
                st.error(
                    f"Design worker **crashed** (signal **{-code}**). "
                    "See README (Streamlit / SIGSEGV) and the terminal for details."
                )
                if raw:
                    st.code(raw[:4000])
                return

            if not raw:
                st.error(f"Design worker produced empty output (exit {code}). Check the terminal.")
                return

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid worker output ({code=}): {e}")
                st.code(raw[:2000])
                return

            if code != 0 or (isinstance(payload, dict) and payload.get("ok") is False):
                err = payload.get("error", raw) if isinstance(payload, dict) else raw
                st.error("Design worker failed.")
                st.code(str(err)[:8000])
                return
            rows = payload
            if not isinstance(rows, list):
                st.error("Unexpected worker result (expected a list).")
                st.json(payload if isinstance(payload, dict) else {"raw": raw[:500]})
                return
        except Exception as e:
            st.exception(e)
            return
        finally:
            if result_path.is_file():
                result_path.unlink(missing_ok=True)

        selected_for_structure = sum(1 for r in rows if r.get("selected_for_structure"))
        with_structure = sum(1 for r in rows if r.get("structure_pdb"))
        with_ddg = sum(1 for r in rows if r.get("ddg_pred") is not None)
        rescued = sum(1 for r in rows if str(r.get("selected_by", "")).strip() == "rescue_lane")
        with_sasa = sum(
            1
            for r in rows
            if (r.get("physics") or {}).get("sasa_total_area") is not None
        )
        st.session_state["last_petase_run"] = {
            "out_path": str(out_p),
            "run_summary_path": str(out_p.parent / "run_summary.json"),
            "work_root": str(work_root),
            "n_rows": len(rows),
            "n_cycles": int(n_cycles),
            "n_mut": int(n_mut),
            "use_colabfold": bool(use_cf),
            "use_topk": bool(use_topk and use_cf),
            "top_k": int(top_k) if use_topk and use_cf else None,
            "seed": int(seed),
            "selected_for_structure": selected_for_structure,
            "with_structure": with_structure,
            "with_sasa": with_sasa,
            "with_ddg": with_ddg,
            "rescued": rescued,
            "ddg_survivor_pct": float(ddg_survivor_pct) / 100.0,
        }

        st.success(f"Wrote {len(rows)} lines to `{out_p}`")
        summary = _load_run_summary_for_jsonl(out_p)
        if summary:
            _render_run_report_cards(summary)
        best_composite = None
        if rows:
            best_composite = max(
                float((r.get("physics") or {}).get("composite", float("-inf")))
                for r in rows
            )
        best_hybrid = None
        if rows:
            best_hybrid = max(float(r.get("hybrid_score", float("-inf"))) for r in rows)
        structure_rate = (with_structure / len(rows) * 100.0) if rows else 0.0
        sasa_rate = (with_sasa / len(rows) * 100.0) if rows else 0.0
        ddg_rate = (with_ddg / len(rows) * 100.0) if rows else 0.0
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Variants", len(rows))
        c2.metric("Selected for structure", selected_for_structure)
        c3.metric("Structure success", f"{structure_rate:.1f}%")
        c4.metric("SASA coverage", f"{sasa_rate:.1f}%")
        c5.metric("Best composite", "n/a" if best_composite is None else f"{best_composite:.4f}")
        c6.metric("ddG staged", f"{ddg_rate:.1f}%")
        c7.metric("Best hybrid", "n/a" if best_hybrid is None else f"{best_hybrid:.4f}")
        if with_ddg < max(1, int(len(rows) * 0.5)):
            st.caption(
                "Seeing many `not_in_ddg_budget` rows is expected: ddG stage only scores a survivor subset."
            )
        if selected_for_structure > 0 and with_structure == 0:
            st.error(
                "Structure stage selected variants but produced 0 structures. "
                "Check `petase_design_runs/structures/*/colabfold.stderr.log` for ColabFold errors."
            )

        if rows:
            disp = _petase_results_dataframe(rows)
            st.markdown("#### Leaderboard")
            st.caption("Columns prioritize generation, structure, composite score, and SASA / Rg metrics.")
            st.dataframe(
                (
                    disp.sort_values(["pareto_rank", "objective_scalar"], ascending=[True, False]).head(50)
                    if {"pareto_rank", "objective_scalar"}.issubset(set(disp.columns))
                    else (
                        disp.sort_values("hybrid_score", ascending=False).head(50)
                        if "hybrid_score" in disp.columns
                        else disp.sort_values("physics.composite", ascending=False).head(50)
                    )
                ),
                width="stretch",
                height=420,
            )
            _render_phase2_analytics(rows, key_prefix="petase_run")
            _render_variant_detail_drawer(rows)


def _render_structure_workspace(*, key_prefix: str = "predict") -> None:
    """Unified structure workflow (sequence helper + manual PDB upload)."""
    st.markdown("#### Structure Workspace")
    st.caption(
        "Use a sequence or an uploaded PDB to inspect structure. "
        "Known curated sequences load resolved structures automatically."
    )
    with st.expander("Viewer diagnostics", expanded=False):
        if st.checkbox("Show 3D environment diagnostics", key=f"{key_prefix}_struct_diag"):
            st.json(format_py3dmol_diagnostics())
        if st.checkbox("Explain network context + test 3Dmol URL", key=f"{key_prefix}_struct_nethelp"):
            render_3dmol_network_help(key_prefix=f"{key_prefix}_struct_nethelp")

    seq_input = st.text_area(
        "Sequence input",
        height=100,
        placeholder="MKT...",
        key=f"{key_prefix}_sequence_input",
    )
    seq_col1, seq_col2 = st.columns([1, 2.2])
    if seq_col1.button("Build sequence structure", key=f"{key_prefix}_build_seq_model"):
        clean = sanitize_sequence(seq_input)
        if len(clean) < 20:
            st.error("Please provide at least 20 amino acids.")
        else:
            ident = identify_sequence(
                clean,
                petase_wt_fasta=REPO_ROOT / "petase_design" / "data" / "petase_6eqd_chainA_notag.fasta",
            )
            known = find_known_structure_match(clean)
            base = f"{key_prefix}_sequence_helper"
            if known:
                pdb_text, err = fetch_known_structure_pdb(known)
                if pdb_text:
                    ident = {
                        "label": f"{ident.get('label', 'custom_sequence')} + known_structure",
                        "detail": (
                            f"{ident.get('detail', '')} "
                            f"{known.get('match_detail', '')} "
                            f"Using PDB {known.get('pdb_id', '?')} chain {known.get('chain', '?')}."
                        ).strip(),
                    }
                    st.session_state[f"{base}_pdb"] = pdb_text
                    st.session_state[f"{base}_source"] = "known_pdb"
                    st.session_state[f"{base}_source_note"] = (
                        f"Matched **{known.get('name', 'known structure')}** "
                        f"-> PDB **{known.get('pdb_id', '?')}** chain **{known.get('chain', '?')}**."
                    )
                else:
                    st.warning(
                        "Matched a known sequence but could not download its PDB; "
                        f"falling back to pseudo model. ({err})"
                    )
                    st.session_state[f"{base}_pdb"] = build_pseudo_pdb_from_sequence(clean)
                    st.session_state[f"{base}_source"] = "pseudo"
                    st.session_state[f"{base}_source_note"] = (
                        "Pseudo model fallback (known structure download failed)."
                    )
            else:
                st.session_state[f"{base}_pdb"] = build_pseudo_pdb_from_sequence(clean)
                st.session_state[f"{base}_source"] = "pseudo"
                st.session_state[f"{base}_source_note"] = "Pseudo model (no curated known-structure match)."
            st.session_state[f"{base}_ident"] = ident

    base = f"{key_prefix}_sequence_helper"
    ident = st.session_state.get(f"{base}_ident")
    helper_pdb = st.session_state.get(f"{base}_pdb")
    helper_source = st.session_state.get(f"{base}_source", "pseudo")
    helper_source_note = st.session_state.get(f"{base}_source_note", "")
    if ident and isinstance(ident, dict):
        seq_col2.info(f"Sequence ID: **{ident.get('label','unknown')}** — {ident.get('detail','')}")
    if helper_pdb and isinstance(helper_pdb, str):
        if helper_source == "known_pdb":
            st.success(helper_source_note or "Loaded curated known PDB for this sequence.")
        elif helper_source_note:
            st.caption(helper_source_note)
        render_structure_background_motion(helper_pdb, key_prefix=f"{key_prefix}_seq_bg")
        with st.expander("Interactive sequence structure", expanded=True):
            render_structure_panel(
                helper_pdb,
                key_prefix=f"{key_prefix}_seq_full",
                default_style=DEFAULT_VIZ_STYLE,
                default_spin=True,
                height=520,
            )

    st.markdown("#### Upload PDB")
    up = st.file_uploader("Upload PDB file", type=["pdb", "ent"], key=f"{key_prefix}_pdb_upload")
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
        if "ATOM" not in text and "HETATM" not in text:
            st.error("File does not look like PDB (no ATOM/HETATM records).")
            return
        render_structure_panel(
            text,
            key_prefix=f"{key_prefix}_uploaded",
            default_style=DEFAULT_VIZ_STYLE,
            default_spin=DEFAULT_VIZ_SPIN,
            height=500,
        )
        st.download_button(
            "Download uploaded PDB",
            data=text.encode("utf-8"),
            file_name=up.name or "structure.pdb",
            mime="chemical/x-pdb",
            key=f"{key_prefix}_dl_pdb",
        )


def tab_jsonl() -> None:
    st.subheader("JSONL Run Browser")
    path = st.text_input(
        "JSONL path",
        value=str(REPO_ROOT / "petase_design_runs" / "gui_run.jsonl"),
    )
    max_rows = st.number_input("Max rows to load", min_value=10, max_value=50_000, value=2000)

    if st.button("Load", key="jsonl_load_btn"):
        p = Path(path)
        if not p.is_file():
            st.error("File not found.")
            return
        rows: list[dict] = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            st.warning("No rows parsed.")
            return
        st.session_state["jsonl_loaded_rows"] = rows
        st.session_state["jsonl_loaded_path"] = str(p.resolve())
        st.session_state["jsonl_loaded_summary"] = _load_run_summary_for_jsonl(p)

    rows = st.session_state.get("jsonl_loaded_rows")
    loaded_path = st.session_state.get("jsonl_loaded_path")
    if rows and loaded_path:
        st.caption(f"Loaded file: `{loaded_path}`")
        if st.button("Clear loaded data", key="jsonl_clear_btn"):
            for k in ("jsonl_loaded_rows", "jsonl_loaded_path", "jsonl_loaded_summary"):
                st.session_state.pop(k, None)
            st.rerun()
        summary = st.session_state.get("jsonl_loaded_summary")
        if summary:
            _render_run_report_cards(summary)
        df = _petase_results_dataframe(rows)
        st.dataframe(df, width="stretch", height=400)

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        sort_col = st.selectbox(
            "Sort by",
            options=numeric_cols if numeric_cols else list(df.columns),
            key="jsonl_sort_col",
        )
        if sort_col in df.columns:
            st.dataframe(
                df.sort_values(sort_col, ascending=False).head(100),
                width="stretch",
            )
        _render_phase2_analytics(rows, key_prefix="jsonl_browse")
        with st.expander("Run comparison (top-k overlap)", expanded=False):
            path_b = st.text_input(
                "Second JSONL path",
                value="",
                key="jsonl_compare_path_b",
            )
            top_k_cmp = st.number_input(
                "Top-K for overlap",
                min_value=1,
                max_value=1000,
                value=20,
                key="jsonl_compare_topk",
            )
            if st.button("Compare runs", key="jsonl_compare_btn"):
                p_b = Path(path_b)
                if not p_b.is_file():
                    st.error("Second JSONL file not found.")
                else:
                    rows_b: list[dict] = []
                    with p_b.open(encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= int(max_rows):
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rows_b.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                    if not rows_b:
                        st.error("No rows parsed from second file.")
                    else:
                        def _sort_rows(rs: list[dict]) -> list[dict]:
                            if rs and rs[0].get("pareto_rank") is not None:
                                return sorted(
                                    rs,
                                    key=lambda r: (
                                        int(r.get("pareto_rank") or 9999),
                                        -float(r.get("objective_scalar") or -1e9),
                                    ),
                                )
                            return sorted(rs, key=lambda r: -float(r.get("hybrid_score") or -1e9))

                        a_top = [
                            str(r.get("job_id", ""))
                            for r in _sort_rows(rows)[: int(top_k_cmp)]
                            if str(r.get("job_id", "")).strip()
                        ]
                        b_top = [
                            str(r.get("job_id", ""))
                            for r in _sort_rows(rows_b)[: int(top_k_cmp)]
                            if str(r.get("job_id", "")).strip()
                        ]
                        sa, sb = set(a_top), set(b_top)
                        inter = len(sa & sb)
                        union = max(1, len(sa | sb))
                        jacc = inter / union
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Top-K overlap", inter)
                        c2.metric("Jaccard", f"{jacc:.3f}")
                        c3.metric("Shared % of K", f"{(100.0 * inter / max(int(top_k_cmp), 1)):.1f}%")
                        st.caption(f"Shared IDs: {sorted(sa & sb)}")


def tab_design_prediction() -> None:
    """Unified workspace: single-sequence analysis + PETase batch design."""
    st.markdown("### Unified Workflow")
    st.caption(
        "Use single-sequence analysis for focused ddG/structure inspection, "
        "then run PETase batch design with hybrid reranking for candidate discovery."
    )
    with st.expander("A) Single-sequence analysis", expanded=True):
        tab_predict()
    st.markdown("---")
    with st.expander("B) PETase batch design", expanded=True):
        tab_petase()


def main() -> None:
    st.set_page_config(
        page_title="petase-thermostability-benchmark",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_presentation_css()
    st.title("petase-thermostability-benchmark")
    st.markdown(
        f"Working directory for file paths: `{REPO_ROOT}`. "
        "Model and FASTA paths are easiest when provided relative to this folder."
    )

    tab1, tab2 = st.tabs(["Design + Prediction", "Browse JSONL"])
    with tab1:
        tab_design_prediction()
    with tab2:
        tab_jsonl()


if __name__ == "__main__":
    main()
