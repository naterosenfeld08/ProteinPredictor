"""
ProteinPredictor — browser GUI (Streamlit).

Run from the repo root:
    streamlit run gui/app.py

You can also double-click a small shell script or use Automator; the app opens in your default browser.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
    identify_sequence,
    sanitize_sequence,
)
from gui.structure_view import render_structure_background_motion, render_structure_panel


def _apply_presentation_css(enabled: bool) -> None:
    if not enabled:
        return
    st.markdown(
        """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
h1, h2, h3 {letter-spacing: 0.2px;}
div[data-testid="stMetric"] {
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 12px;
  padding: 0.55rem 0.7rem;
  background: rgba(20,20,30,0.2);
}
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
    slot.empty()
    return int(proc.returncode or 0)


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
    st.subheader("Predict ΔΔG (stability change)")
    render_fireprot_honesty_callout()
    st.caption(
        "Uses your trained ensemble + PLM embeddings. First run may download "
        "ProtT5/ESM weights and can take several minutes."
    )
    last = st.session_state.get("last_prediction")
    if last:
        with st.expander("Last finished prediction (this browser session)", expanded=False):
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


def _petase_results_dataframe(rows: list) -> pd.DataFrame:
    """Column order: key structure / SASA / composite fields first, then the rest."""
    df = pd.json_normalize(rows)
    priority = [
        "generation",
        "selected_for_structure",
        "structure_pdb",
        "physics.composite",
        "physics.sasa_total_area",
        "physics.apolar_sasa_fraction",
        "physics.radius_of_gyration",
        "physics.mutation_count",
        "physics.active_site_violation",
        "physics.mean_hydrophobicity",
        "physics.net_charge_proxy",
    ]
    head = [c for c in priority if c in df.columns]
    tail = [c for c in df.columns if c not in head]
    return df[head + tail]


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


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
    struct_pct = (100.0 * n_struct / n_variants) if n_variants else 0.0
    sasa_pct = (100.0 * n_sasa / n_variants) if n_variants else 0.0
    runtime_s = _safe_float(runtime.get("seconds_wall")) or 0.0

    st.markdown("#### Run report cards")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Variants", n_variants)
    c2.metric("With structure", f"{n_struct} ({struct_pct:.1f}%)")
    c3.metric("With SASA", f"{n_sasa} ({sasa_pct:.1f}%)")
    c4.metric("Runtime (s)", f"{runtime_s:.3f}")
    c5.metric("Top-K mode", str(run_meta.get("structure_top_k") or "off"))

    top = summary.get("top_variants") or []
    if top:
        st.caption("Top 10 variants from run summary")
        st.dataframe(pd.DataFrame(top), use_container_width=True, height=280)


def _render_phase2_analytics(rows: list[dict], *, key_prefix: str) -> None:
    if not rows:
        return
    df = _petase_results_dataframe(rows)
    if "generation" in df.columns:
        df["generation"] = pd.to_numeric(df["generation"], errors="coerce")

    st.markdown("#### Statistical analytics")
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
                use_container_width=True,
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
                    use_container_width=True,
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
                    use_container_width=True,
                )

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dist_candidates = [c for c in numeric_cols if c.startswith("physics.")]
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
            st.bar_chart(hist_df, height=220, use_container_width=True)

    corr_cols = [c for c in numeric_cols if c.startswith("physics.")]
    if len(corr_cols) >= 2:
        st.caption("Correlation matrix (physics features)")
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        st.dataframe(corr, use_container_width=True, height=260)


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
    st.markdown("#### Pipeline storyboard")
    st.caption("This run flow combines LLM-informed sequence context with structure-aware rescoring.")
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
    st.caption(f"Mutations: {muts_txt}")

    structure_pdb = row.get("structure_pdb")
    if structure_pdb:
        p = Path(str(structure_pdb))
        if p.is_file():
            pdb_text = p.read_text(encoding="utf-8", errors="replace")
            render_structure_panel(
                pdb_text,
                key_prefix=f"drawer_{selected}",
                default_style=st.session_state.get("viz_style_default", "cartoon_amino"),
                default_spin=bool(st.session_state.get("viz_spin_default", False)),
                height=560 if st.session_state.get("presentation_mode", False) else 500,
            )
        else:
            st.warning(f"Structure path not found on disk: `{p}`")


def tab_petase() -> None:
    st.subheader("PETase Design Studio")
    st.caption("Mission control for variant generation, structure rescoring, and rapid triage.")
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
        st.caption(
            "ColabFold progress streams in terminal while this panel tracks pipeline status."
        )

    if run_clicked:
        wt_p = Path(wt)
        if not wt_p.is_file():
            st.error(f"WT FASTA not found: {wt_p}")
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

        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
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
        structure_rate = (with_structure / len(rows) * 100.0) if rows else 0.0
        sasa_rate = (with_sasa / len(rows) * 100.0) if rows else 0.0
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Variants", len(rows))
        c2.metric("Selected for structure", selected_for_structure)
        c3.metric("Structure success", f"{structure_rate:.1f}%")
        c4.metric("SASA coverage", f"{sasa_rate:.1f}%")
        c5.metric("Best composite", "n/a" if best_composite is None else f"{best_composite:.4f}")
        c6.metric("ColabFold", "on" if use_cf else "off")

        if rows:
            disp = _petase_results_dataframe(rows)
            st.markdown("#### Leaderboard")
            st.caption("Columns ordered: generation -> structure -> composite -> SASA / Rg -> ...")
            st.dataframe(
                disp.sort_values("physics.composite", ascending=False).head(50),
                use_container_width=True,
                height=420,
            )
            _render_phase2_analytics(rows, key_prefix="petase_run")
            _render_variant_detail_drawer(rows)


def tab_structure() -> None:
    st.subheader("Structure viewer (optional)")
    st.caption(
        "**PyMOL** is a separate desktop app; this tab gives you an **in-browser** preview (py3Dmol) "
        "plus a small **PyMOL script** you can run locally. It does **not** change ΔΔG predictions."
    )
    st.markdown("#### Sequence visual helper")
    seq_input = st.text_area(
        "Paste amino acid sequence for quick visual model",
        height=110,
        placeholder="MKT...",
        key="structure_seq_input",
    )
    seq_col1, seq_col2 = st.columns([1, 2.2])
    if seq_col1.button("Build sequence model", key="build_seq_model"):
        clean = sanitize_sequence(seq_input)
        if len(clean) < 20:
            st.error("Please provide at least 20 amino acids.")
        else:
            ident = identify_sequence(
                clean,
                petase_wt_fasta=REPO_ROOT / "petase_design" / "data" / "petase_6eqd_chainA_notag.fasta",
            )
            st.session_state["sequence_helper_ident"] = ident
            st.session_state["sequence_helper_pdb"] = build_pseudo_pdb_from_sequence(clean)
    ident = st.session_state.get("sequence_helper_ident")
    helper_pdb = st.session_state.get("sequence_helper_pdb")
    if ident and isinstance(ident, dict):
        seq_col2.info(f"Sequence ID: **{ident.get('label','unknown')}** — {ident.get('detail','')}")
    if helper_pdb and isinstance(helper_pdb, str):
        st.caption("Soft-motion background preview")
        render_structure_background_motion(helper_pdb, key_prefix="seq_helper_bg")
        with st.expander("Full interactive sequence model", expanded=True):
            render_structure_panel(
                helper_pdb,
                key_prefix="seq_helper_full",
                default_style=st.session_state.get("viz_style_default", "cartoon_amino"),
                default_spin=True,
                height=620 if st.session_state.get("presentation_mode", False) else 520,
            )

    st.markdown("#### Upload real PDB")
    up = st.file_uploader("Upload PDB", type=["pdb", "ent"], key="pdb_upload")
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
        if "ATOM" not in text and "HETATM" not in text:
            st.error("File does not look like PDB (no ATOM/HETATM records).")
            return
        render_structure_panel(
            text,
            key_prefix="main",
            default_style=st.session_state.get("viz_style_default", "cartoon_amino"),
            default_spin=bool(st.session_state.get("viz_spin_default", False)),
            height=620 if st.session_state.get("presentation_mode", False) else 500,
        )
        st.download_button(
            "Download uploaded PDB",
            data=text.encode("utf-8"),
            file_name=up.name or "structure.pdb",
            mime="chemical/x-pdb",
            key="dl_pdb",
        )


def tab_jsonl() -> None:
    st.subheader("Browse a design JSONL log")
    path = st.text_input(
        "JSONL path",
        value=str(REPO_ROOT / "petase_design_runs" / "gui_run.jsonl"),
    )
    max_rows = st.number_input("Max rows to load", min_value=10, max_value=50_000, value=2000)

    if st.button("Load"):
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
        summary = _load_run_summary_for_jsonl(p)
        if summary:
            _render_run_report_cards(summary)
        df = _petase_results_dataframe(rows)
        st.dataframe(df, use_container_width=True, height=400)

        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        sort_col = st.selectbox(
            "Sort by",
            options=numeric_cols if numeric_cols else list(df.columns),
        )
        if sort_col in df.columns:
            st.dataframe(
                df.sort_values(sort_col, ascending=False).head(100),
                use_container_width=True,
            )
        _render_phase2_analytics(rows, key_prefix="jsonl_browse")


def main() -> None:
    st.set_page_config(
        page_title="ProteinPredictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("ProteinPredictor")
    with st.sidebar:
        st.markdown("### View options")
        st.session_state["presentation_mode"] = st.checkbox(
            "Presentation mode",
            value=bool(st.session_state.get("presentation_mode", False)),
            help="Bigger visuals and cleaner cards for demos.",
        )
        st.session_state["viz_style_default"] = st.selectbox(
            "Default structure preset",
            options=["cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"],
            index=["cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"].index(
                st.session_state.get("viz_style_default", "cartoon_amino")
            )
            if st.session_state.get("viz_style_default", "cartoon_amino")
            in {"cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"}
            else 0,
        )
        st.session_state["viz_spin_default"] = st.checkbox(
            "Default auto-rotate structures",
            value=bool(st.session_state.get("viz_spin_default", False)),
        )
    _apply_presentation_css(bool(st.session_state.get("presentation_mode", False)))
    st.markdown(
        f"Working directory for paths: `{REPO_ROOT}` — run Streamlit from anywhere, "
        "but model/FASTA paths are easiest if relative to this folder."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["ΔΔG predict", "PETase design", "Browse JSONL", "Structure"]
    )
    with tab1:
        tab_predict()
    with tab2:
        tab_petase()
    with tab3:
        tab_jsonl()
    with tab4:
        tab_structure()


if __name__ == "__main__":
    main()
