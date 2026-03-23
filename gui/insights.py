"""
Streamlit helpers: interpret ΔΔG outputs with FireProt context and honest limits.
"""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

# From README / FireProtDB summary (training distribution — not ground truth for arbitrary proteins).
FIREPROT_DDG_MEAN_KCALMOL = 0.5
FIREPROT_DDG_STD_KCALMOL = 2.0
FIREPROT_DDG_TYPICAL_MIN = -10.0
FIREPROT_DDG_TYPICAL_MAX = 10.0


def render_fireprot_honesty_callout() -> None:
    st.info(
        "**Science note — FireProt ΔΔG model:** This regressor was trained on **single-point "
        "mutation ΔΔG** measurements in **many proteins** (FireProtDB), **not** on PETase-specific "
        "mutants. For a pasted sequence it returns a **scalar in kcal/mol on that training scale** — "
        "useful as a **weak, generic stability prior** for screening or ranking with **fixed protocol**, "
        "not as an experimental ΔΔG or melting temperature. For PET engineering, treat it as **non–PET-specific** "
        "unless you **retrain or fine-tune** on PETase mutant data."
    )


def render_prediction_analytics(p0: Mapping[str, Any], *, seq_name: str, seq_len: int) -> None:
    """Richer post-prediction panel: z-score vs training stats, uncertainty, interval logic."""
    pred = float(p0["pred_value"])
    unc_raw = p0.get("uncertainty")
    unc = float(unc_raw) if unc_raw is not None else None

    st.subheader("Interpretation & statistics")
    c1, c2, c3 = st.columns(3)
    with c1:
        z = (pred - FIREPROT_DDG_MEAN_KCALMOL) / FIREPROT_DDG_STD_KCALMOL
        st.metric(
            "Z-score vs FireProt mean",
            f"{z:.2f}",
            help=f"Using training-summary mean={FIREPROT_DDG_MEAN_KCALMOL}, "
            f"std≈{FIREPROT_DDG_STD_KCALMOL} kcal/mol (approximate).",
        )
        st.caption("~0 = near typical FireProt mutation; |z|>2 = unusually high/low on that scale.")
    with c2:
        if unc is not None and unc > 0:
            rel = abs(pred) / unc
            st.metric("|prediction| / σ (trees)", f"{rel:.2f}")
            st.caption("Low values → magnitude is small vs RF disagreement; don't trust the sign tightly.")
        else:
            st.metric("|prediction| / σ (trees)", "—")
            st.caption("No tree-variance uncertainty for this model path.")
    with c3:
        frac_span = None
        if unc is not None and unc > 0:
            width = 2 * 1.96 * unc
            frac_span = width / (FIREPROT_DDG_TYPICAL_MAX - FIREPROT_DDG_TYPICAL_MIN)
            st.metric("95% interval width / ~FireProt span", f"{frac_span:.2f}")
            st.caption("~1+ means the interval is as wide as the rough ±10 kcal/mol literature range.")
        else:
            st.metric("Interval width / span", "—")

    lo = hi = None
    if unc is not None:
        lo = pred - 1.96 * unc
        hi = pred + 1.96 * unc

    with st.expander("Uncertainty & error analysis (read this)", expanded=False):
        st.markdown(
            f"- **Point estimate:** {pred:.4f} kcal/mol  \n"
            f"- **Sequence:** `{seq_name}` ({seq_len} aa)  \n"
        )
        if unc is not None:
            st.markdown(
                f"- **RF σ (across trees):** {unc:.4f} kcal/mol — not experimental error; "
                "trees disagree when the embedding is ambiguous for this task.  \n"
                f"- **Approx. 95% interval (normal approx.):** [{lo:.4f}, {hi:.4f}] — "
                "assumes tree spread is Gaussian; use as a **rough** screen only.  \n"
            )
            crosses = lo <= 0 <= hi
            st.markdown(
                "- **Interval crosses 0:** "
                + ("**Yes** — sign of ΔΔG is not well determined." if crosses else "**No** — model leans one sign, still not lab truth.")
            )
        else:
            st.markdown("- No per-tree uncertainty returned for this checkpoint.")

        st.markdown(
            "\n**Common misinterpretations to avoid:**\n"
            "- Treating the number as **measured** ΔΔG for this exact sequence in buffer.  \n"
            "- Using it for **PET activity** or **Tm** — different experiments.  \n"
            "- Comparing to literature **without** the same mutation definition (WT vs mutant pair)."
        )

    # Position within coarse training range
    clamped = max(FIREPROT_DDG_TYPICAL_MIN, min(FIREPROT_DDG_TYPICAL_MAX, pred))
    pos = (clamped - FIREPROT_DDG_TYPICAL_MIN) / (
        FIREPROT_DDG_TYPICAL_MAX - FIREPROT_DDG_TYPICAL_MIN
    )
    st.progress(
        pos,
        text=f"Rough position in ~[{FIREPROT_DDG_TYPICAL_MIN:.0f}, {FIREPROT_DDG_TYPICAL_MAX:.0f}] kcal/mol training scale "
        f"(clipped display; prediction={pred:.3f})",
    )
    if not (FIREPROT_DDG_TYPICAL_MIN <= pred <= FIREPROT_DDG_TYPICAL_MAX):
        st.caption(
            f"Prediction is **outside** the coarse ±10 kcal/mol window used for this bar — "
            f"still possible on the learned scale; interpret cautiously."
        )
