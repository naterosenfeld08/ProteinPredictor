"""
Optional in-browser structure (py3Dmol) + PyMOL script download for desktop viewing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

_DEFAULT_3DMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"


def _py3dmol_js_file_path() -> Path | None:
    raw = os.environ.get("PY3DMOL_JS_FILE", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def _py3dmol_js_url_for_view() -> str:
    """
    URL passed to py3Dmol's ``js=`` parameter (wires ``loadScriptAsync`` in generated HTML).

    **Important:** Chrome blocks ``<script src="data:...">`` inside Streamlit's sandboxed iframe,
    so we never use a data URI here. For local copies, use ``PY3DMOL_JS_FILE`` and
    :func:`_patch_py3dmol_html_inline_library` instead.

    - ``PY3DMOL_JS_URL`` — https URL to 3Dmol-min.js (mirror)
    - else default jsDelivr
    """
    url = os.environ.get("PY3DMOL_JS_URL", "").strip()
    if url:
        return url
    return _DEFAULT_3DMOL_CDN


def _patch_py3dmol_html_inline_library(html: str, js_file: Path) -> str:
    """
    Prepend inline 3Dmol.js and skip async CDN load (Chrome / iframe safe).
    """
    raw = js_file.read_text(encoding="utf-8", errors="replace")
    safe = raw.replace("</script>", "<\\/script>")
    prefix = f'<script type="text/javascript">\n{safe}\n</script>\n'
    html2, n = re.subn(
        r"\$3Dmolpromise\s*=\s*loadScriptAsync\([^)]*\)\s*;",
        "$3Dmolpromise = Promise.resolve();",
        html,
        count=1,
    )
    if n == 0:
        st.warning(
            "Could not patch py3Dmol loader for inline 3Dmol.js — viewer may stay blank. "
            "Try unsetting PY3DMOL_JS_FILE and using the default CDN, or upgrade py3Dmol."
        )
    return prefix + html2


def _finalize_py3dmol_html(html: str) -> str:
    js_path = _py3dmol_js_file_path()
    if js_path is not None:
        return _patch_py3dmol_html_inline_library(html, js_path)
    return html


def format_py3dmol_diagnostics() -> dict[str, str]:
    """Safe one-screen debug payload for the Structure tab."""
    out: dict[str, str] = {}
    try:
        import py3Dmol  # type: ignore[import-not-found]

        out["py3Dmol_version"] = str(getattr(py3Dmol, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        out["py3Dmol_version"] = f"import failed: {type(e).__name__}: {e}"
    raw_file = os.environ.get("PY3DMOL_JS_FILE", "").strip()
    if raw_file and _py3dmol_js_file_path() is None:
        out["PY3DMOL_JS_FILE_warning"] = f"path not found or not a file: {raw_file!r}"

    fpath = _py3dmol_js_file_path()
    if fpath is not None:
        out["3dmol_js_mode"] = "inline file (prepended before py3Dmol bootstrap; Chrome-safe)"
        out["PY3DMOL_JS_FILE_resolved"] = str(fpath.resolve())
    else:
        out["3dmol_js_mode"] = "async URL (py3Dmol loadScriptAsync)"
        out["3dmol_js_url"] = _py3dmol_js_url_for_view()
    out["PY3DMOL_JS_FILE"] = os.environ.get("PY3DMOL_JS_FILE", "") or "(not set)"
    out["PY3DMOL_JS_URL"] = os.environ.get("PY3DMOL_JS_URL", "") or "(not set)"
    return out


def _view_to_html(view: object) -> str | None:
    if hasattr(view, "_make_html"):
        html = view._make_html()
        if isinstance(html, str) and html.strip():
            return html
    if hasattr(view, "_repr_html_"):
        html = view._repr_html_()
        if isinstance(html, str) and html.strip():
            return html
    return None


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


def render_structure_background_motion(
    pdb_text: str,
    *,
    key_prefix: str = "bg",
    height: int = 220,
) -> None:
    """
    Render a soft-motion structure as a visual backdrop panel.

    Note: Do not apply CSS ``filter: blur()`` on the viewer container; it breaks WebGL in many browsers.
    """
    try:
        import py3Dmol  # type: ignore[import-not-found]
        import streamlit.components.v1 as components
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Background structure view unavailable ({type(exc).__name__}: {exc}).")
        return

    js = _py3dmol_js_url_for_view()
    view = py3Dmol.view(width=980, height=height, js=js)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.spin(True)
    view.zoomTo()
    inner = _view_to_html(view)
    if not inner:
        st.error("Could not generate py3Dmol HTML for background preview.")
        return
    inner = _finalize_py3dmol_html(inner)
    wrapped = f"""
<div style="position: relative; width: 100%; border-radius: 14px; overflow: hidden; opacity: 0.88;">
  {inner}
  <div style="position:absolute; inset:0; pointer-events:none; background: linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.18));"></div>
</div>
"""
    components.html(wrapped, width=1000, height=height + 20, scrolling=False)


def render_structure_panel(
    pdb_text: str,
    *,
    key_prefix: str = "struct",
    show_controls: bool = True,
    default_style: str = "cartoon_amino",
    default_spin: bool = False,
    height: int = 500,
    show_troubleshoot_caption: bool = True,
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

        js = _py3dmol_js_url_for_view()
        view = py3Dmol.view(width=900, height=height, js=js)
        view.addModel(pdb_text, "pdb")
        _apply_style(view, style)
        if spin:
            view.spin(True)
        view.zoomTo()
        html = _view_to_html(view)
        if html:
            html = _finalize_py3dmol_html(html)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not build in-browser viewer ({type(exc).__name__}: {exc}). "
            "`pip install py3dmol` in the same environment as Streamlit, then refresh."
        )

    if html:
        import streamlit.components.v1 as components

        components.html(html, width=920, height=height + 20, scrolling=False)
        if show_troubleshoot_caption:
            n_atom = sum(
                1
                for line in pdb_text.splitlines()
                if line.startswith("ATOM") or line.startswith("HETATM")
            )
            st.caption(
                f"Viewer HTML generated ({len(html)} chars, ~{n_atom} ATOM lines). "
                "If the canvas stays blank: **DevTools → Console** for WebGL/script errors; **Network** for `3Dmol-min.js` "
                "(unless `PY3DMOL_JS_FILE` inlines it). **Do not** use a `data:` URL for `js=` — Chrome blocks it in iframes; "
                "use `PY3DMOL_JS_FILE` (now inlined) or `PY3DMOL_JS_URL` / default CDN."
            )
    elif show_troubleshoot_caption:
        st.error(
            "py3Dmol did not return HTML (unexpected). Check that `py3Dmol` is installed and the PDB text contains ATOM records."
        )

    st.download_button(
        label="Download PyMOL script (load_view.pml)",
        data=pymol_load_script("uploaded.pdb"),
        file_name="load_view.pml",
        mime="text/plain",
        key=f"{key_prefix}_pml",
        help="Save your PDB as uploaded.pdb in the same folder, or edit the load line inside the script.",
    )
