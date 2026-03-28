"""
Optional in-browser structure (py3Dmol) + PyMOL script download for desktop viewing.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import tempfile
import threading
from pathlib import Path

import streamlit as st

_DEFAULT_3DMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js"

_local_3dmol_lock = threading.Lock()
_local_3dmol_cache: dict[str, str] = {}


def _serve_3dmol_min_js_on_loopback(js_path: Path) -> str:
    """
    Streamlit forwards ``components.html`` via WebSocket; multi‑MB ``srcdoc`` payloads are unreliable.
    Serve a local copy of ``3Dmol-min.js`` on 127.0.0.1 so the iframe stays small and loads the lib via HTTP.
    """
    key = str(js_path.resolve())
    with _local_3dmol_lock:
        hit = _local_3dmol_cache.get(key)
        if hit:
            return hit
        d = tempfile.mkdtemp(prefix="pp_3dmol_")
        dest = Path(d) / "3Dmol-min.js"
        shutil.copy2(js_path, dest)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        port = int(httpd.server_address[1])
        thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="pp-3dmol-http")
        thread.start()
        url = f"http://127.0.0.1:{port}/3Dmol-min.js"
        _local_3dmol_cache[key] = url
        return url


def _py3dmol_js_file_path() -> Path | None:
    raw = os.environ.get("PY3DMOL_JS_FILE", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def _py3dmol_js_url_for_view() -> str:
    """
    URL passed to py3Dmol's ``js=`` parameter (wires ``loadScriptAsync`` in generated HTML).

    Resolution order:

    - ``PY3DMOL_JS_URL`` — explicit https URL (mirror / corp proxy).
    - ``PY3DMOL_JS_FILE`` — local file is **served on 127.0.0.1** (avoids huge ``srcdoc`` and blocked ``data:`` URLs).
    - else default jsDelivr CDN.
    """
    url = os.environ.get("PY3DMOL_JS_URL", "").strip()
    if url:
        return url
    fpath = _py3dmol_js_file_path()
    if fpath is not None:
        return _serve_3dmol_min_js_on_loopback(fpath)
    return _DEFAULT_3DMOL_CDN


def effective_3dmol_js_url() -> str:
    """Resolved URL passed to py3Dmol (starts loopback server if ``PY3DMOL_JS_FILE`` is set)."""
    return _py3dmol_js_url_for_view()


def probe_3dmol_js_url(url: str, *, read_bytes: int = 256) -> tuple[bool, str]:
    """HTTP GET from the Streamlit server process (same machine as the loopback 3Dmol server)."""
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ProteinPredictor-GUI/1"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            chunk = resp.read(read_bytes)
            code = getattr(resp, "status", None) or resp.getcode()
        return True, f"HTTP {code}, read {len(chunk)} bytes (library reachable from Python)."
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def render_3dmol_network_help(*, key_prefix: str = "nethelp") -> None:
    """Explain why DevTools Network may not list 3Dmol.js on the top document."""
    st.markdown(
        "**Why Network only shows `image/svg+xml`:** that list is usually for the **main Streamlit page** "
        "(icons, UI). The molecular viewer loads **inside nested iframes** (`about:srcdoc`). "
        "Subframe requests are easy to miss unless you switch DevTools context.\n\n"
        "**Chrome:** open DevTools → **Network** → use the **frame / context menu** at the top "
        "(often shows **top** — try the **iframe** entries), or **right‑click** inside the blank viewer → **Inspect**, "
        "then in **Network** with that node selected, reload. You can also use **Application → Frames**.\n\n"
        "Below: the exact **3Dmol.js URL** this app uses, a **server-side fetch check** (from Python), and a link to open it."
    )
    url = effective_3dmol_js_url()
    st.code(url, language="text")
    ok, msg = probe_3dmol_js_url(url)
    if ok:
        st.success(msg)
    else:
        st.error(msg)
    if url.startswith("http://") or url.startswith("https://"):
        st.link_button(
            "Open 3Dmol.js URL in new tab (expect minified JavaScript)",
            url,
            help="If this fails or downloads nothing, the browser cannot reach the library.",
            type="secondary",
            key=f"{key_prefix}_open_js",
        )


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
        out["3dmol_js_mode"] = "local file served on 127.0.0.1 (small iframe srcdoc)"
        out["PY3DMOL_JS_FILE_resolved"] = str(fpath.resolve())
        out["3dmol_effective_js_url"] = _py3dmol_js_url_for_view()
    else:
        out["3dmol_js_mode"] = "async URL (py3Dmol loadScriptAsync)"
        out["3dmol_js_url"] = _py3dmol_js_url_for_view()
    eff = _py3dmol_js_url_for_view()
    ok, probe = probe_3dmol_js_url(eff)
    out["python_probe_3dmol_js"] = probe if ok else f"FAIL: {probe}"
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


def build_standalone_viewer_html(
    pdb_text: str,
    *,
    width: int = 900,
    height: int = 600,
    js_url: str | None = None,
) -> str:
    """
    Full HTML document using the public CDN for 3Dmol.js — open in a normal browser tab if the Streamlit embed fails.
    """
    try:
        import py3Dmol  # type: ignore[import-not-found]
    except ImportError:
        return ""
    js = js_url or _DEFAULT_3DMOL_CDN
    view = py3Dmol.view(width=width, height=height, js=js)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"colorscheme": "amino"}})
    view.zoomTo()
    inner = _view_to_html(view)
    if not inner:
        return ""
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        "<title>ProteinPredictor — structure viewer</title>\n"
        "</head>\n"
        f'<body style="margin:0;background:#1a1c24;">{inner}</body>\n</html>\n'
    )


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
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not build in-browser viewer ({type(exc).__name__}: {exc}). "
            "`pip install py3dmol` in the same environment as Streamlit, then refresh."
        )

    if html:
        import streamlit.components.v1 as components

        components.html(html, width=920, height=height + 20, scrolling=False)
        standalone = build_standalone_viewer_html(pdb_text, width=900, height=height)
        if standalone:
            st.download_button(
                label="Download standalone viewer (open in Chrome)",
                data=standalone.encode("utf-8"),
                file_name="structure_viewer_standalone.html",
                mime="text/html",
                key=f"{key_prefix}_standalone_html",
                help="Uses jsDelivr for 3Dmol.js; works outside Streamlit if the in-app embed is blank.",
            )
        if show_troubleshoot_caption:
            n_atom = sum(
                1
                for line in pdb_text.splitlines()
                if line.startswith("ATOM") or line.startswith("HETATM")
            )
            st.caption(
                f"Viewer HTML generated ({len(html)} chars, ~{n_atom} ATOM/HETATM lines). "
                "If blank: **Network** must load `3Dmol-min.js` (CDN, `PY3DMOL_JS_URL`, or `http://127.0.0.1:…` when "
                "`PY3DMOL_JS_FILE` is set). Huge inlined scripts inside Streamlit’s iframe are avoided on purpose."
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
