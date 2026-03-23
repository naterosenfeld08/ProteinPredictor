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


def render_structure_panel(pdb_text: str, *, key_prefix: str = "struct") -> None:
    """Show py3Dmol if available; always offer PyMOL script download."""
    st.caption(
        "**In-browser:** uses py3Dmol (no PyMOL license needed). "
        "**Desktop PyMOL:** download the `.pml` script and your `.pdb` file."
    )

    html = None
    try:
        import py3Dmol  # type: ignore[import-not-found]

        view = py3Dmol.view(width=720, height=480)
        view.addModel(pdb_text, "pdb")
        view.setStyle({"cartoon": {"colorscheme": "amino"}})
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

        components.html(html, width=740, height=500, scrolling=False)

    st.download_button(
        label="Download PyMOL script (load_view.pml)",
        data=pymol_load_script("uploaded.pdb"),
        file_name="load_view.pml",
        mime="text/plain",
        key=f"{key_prefix}_pml",
        help="Save your PDB as uploaded.pdb in the same folder, or edit the load line inside the script.",
    )
