"""
Optional in-browser structure (py3Dmol) + PyMOL script download for desktop viewing.
"""

from __future__ import annotations

import streamlit as st


def pymol_load_script(pdb_filename: str = "structure.pdb") -> str:
    """User saves uploaded PDB as `pdb_filename` next to this script, then `@load_view.pml` in PyMOL."""
    return f"""# PyMOL — save as load_view.pml next to your PDB ({pdb_filename})
# Open PyMOL → File → Run Script… → select this file
load {pdb_filename}, prot
hide everything
show cartoon, prot
color spectrum, prot
orient prot
zoom prot, buffer=8
"""


def _apply_style(view: object, style: str) -> None:
    if style == "surface":
        view.setStyle({"surface": {"opacity": 0.78, "color": "lightgrey"}})
        return
    if style == "cartoon_sticks":
        view.setStyle({"cartoon": {"colorscheme": "amino"}})
        view.addStyle({"hetflag": False}, {"stick": {"radius": 0.18, "colorscheme": "greenCarbon"}})
        return
    if style == "cartoon_chain":
        view.setStyle({"cartoon": {"color": "spectrum"}})
        return
    # default
    view.setStyle({"cartoon": {"colorscheme": "amino"}})


def render_structure_panel(
    pdb_text: str,
    *,
    key_prefix: str = "struct",
    show_controls: bool = True,
    default_style: str = "cartoon_amino",
    default_spin: bool = False,
    height: int = 500,
) -> None:
    """Show py3Dmol if available; always offer PyMOL script download."""
    st.caption(
        "**In-browser:** uses py3Dmol (no PyMOL license needed). "
        "**Desktop PyMOL:** download the `.pml` script and your `.pdb` file."
    )
    style = default_style
    spin = default_spin
    if show_controls:
        c1, c2 = st.columns([1.7, 1.0])
        style = c1.selectbox(
            "Representation preset",
            options=["cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"],
            index=max(0, ["cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"].index(default_style))
            if default_style in {"cartoon_amino", "cartoon_chain", "cartoon_sticks", "surface"}
            else 0,
            key=f"{key_prefix}_style",
        )
        spin = c2.checkbox("Auto-rotate", value=default_spin, key=f"{key_prefix}_spin")

    html = None
    try:
        import py3Dmol  # type: ignore[import-not-found]

        view = py3Dmol.view(width=900, height=height)
        view.addModel(pdb_text, "pdb")
        _apply_style(view, style)
        if spin:
            view.spin(True)
        view.zoomTo()
        if hasattr(view, "_make_html"):
            html = view._make_html()
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not build in-browser viewer ({type(exc).__name__}: {exc}). "
            "`pip install py3dmol` in the same environment as Streamlit, then refresh."
        )

    if html:
        import streamlit.components.v1 as components

        components.html(html, width=920, height=height + 20, scrolling=False)

    st.download_button(
        label="Download PyMOL script (load_view.pml)",
        data=pymol_load_script("uploaded.pdb"),
        file_name="load_view.pml",
        mime="text/plain",
        key=f"{key_prefix}_pml",
        help="Save your PDB as uploaded.pdb in the same folder, or edit the load line inside the script.",
    )
