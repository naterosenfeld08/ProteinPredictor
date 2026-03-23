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
from gui.structure_view import render_structure_panel


def _poll_subprocess_with_ui(
    proc: subprocess.Popen,
    *,
    title: str,
    detail: str,
    hint_terminal: str = "",
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
        slot.info(
            f"**{title}**{dots}  \n"
            f"{detail}  \n\n"
            f"Elapsed **{elapsed:.0f}s**. First run may download large model weights "
            f"(**10–20+ minutes**).  \n\n"
            f"**Leave this tab open** and check the **terminal** where you ran "
            f"`streamlit run` for download / embedding progress.{extra}"
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


def tab_petase() -> None:
    st.subheader("PETase design loop")
    st.caption(
        "Random mutations from WT FASTA, physics composite score, optional ColabFold "
        "(slow; needs local install + GPU)."
    )

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
    cf_bin = st.text_input("colabfold_batch command", value="colabfold_batch")
    num_recycle = st.number_input("ColabFold num-recycle", min_value=0, max_value=12, value=3)
    use_amber = st.checkbox("ColabFold --amber (OpenMM relax)", value=False)

    if st.button("Run design loop", type="primary"):
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
            if use_amber:
                cmd.append("--amber")

        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            code = _poll_subprocess_with_ui(
                proc,
                title=f"Design loop ({n_cycles} cycles, separate process)",
                detail="Proposing variants and scoring. ColabFold makes this much slower.",
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

        st.success(f"Wrote {len(rows)} lines to `{out_p}`")
        if rows:
            st.dataframe(pd.json_normalize(rows).head(20), use_container_width=True)


def tab_structure() -> None:
    st.subheader("Structure viewer (optional)")
    st.caption(
        "**PyMOL** is a separate desktop app; this tab gives you an **in-browser** preview (py3Dmol) "
        "plus a small **PyMOL script** you can run locally. It does **not** change ΔΔG predictions."
    )
    up = st.file_uploader("Upload PDB", type=["pdb", "ent"], key="pdb_upload")
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
        if "ATOM" not in text and "HETATM" not in text:
            st.error("File does not look like PDB (no ATOM/HETATM records).")
            return
        render_structure_panel(text, key_prefix="main")
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
        df = pd.json_normalize(rows)
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


def main() -> None:
    st.set_page_config(
        page_title="ProteinPredictor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("ProteinPredictor")
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
