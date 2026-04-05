"""
In-browser protein structure (py3Dmol) and optional PyMOL script download for the Streamlit **Structure** tab.

**3Dmol.js loading:** py3Dmol injects a ``<script>`` for the library. By default the app **downloads**
the chosen URL once into the loopback static dir and serves ``/3Dmol-min.js`` from **the same origin**
as ``viewer_….html`` (avoids third‑party / tracking‑prevention blocks on cross‑origin scripts inside
nested iframes). Override behavior:

- ``PY3DMOL_NO_MIRROR=1`` — pass the remote URL straight to py3Dmol (old behavior).
- ``PY3DMOL_JS_URL`` / ``PY3DMOL_JS_FILE`` — same mirror logic unless ``NO_MIRROR``.

Viewer HTML is served over a loopback ``HTTPServer`` and embedded with ``iframe`` ``src=`` (not ``srcdoc``)
so large PDBs and script loading behave reliably in nested frames.
"""

from __future__ import annotations

import http.server
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

# Official Pitt-hosted build (avoids blocked/misrouted jsDelivr in some networks).
# Override with PY3DMOL_JS_URL, e.g. https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js for a pinned mirror.
_DEFAULT_3DMOL_CDN = "https://3dmol.csb.pitt.edu/build/3Dmol-min.js"

# Help nested iframes allocate space; py3Dmol sets px on its div but parent chain can still collapse.
_VIEWER_PAGE_STYLE = (
    "<style>"
    "html,body{margin:0;padding:0;width:100%;height:100%;min-height:100vh;background:#0e1117;}"
    "#pp-outer{min-height:100vh;width:100%;display:block;}"
    "</style>"
)

_servers_lock = threading.Lock()
_servers: dict[str, "_StaticServer"] = {}
_mirror_locks: dict[str, threading.Lock] = {}
_MIN_3DMOL_BYTES = 100_000  # sanity: full min build is ~500kB


def _lock_for_mirror_dir(directory: Path) -> threading.Lock:
    key = str(directory.resolve())
    with _servers_lock:
        if key not in _mirror_locks:
            _mirror_locks[key] = threading.Lock()
        return _mirror_locks[key]


def _mirror_remote_3dmol_to_loopback(srv: _StaticServer, source_url: str) -> str:
    """
    Write ``3Dmol-min.js`` next to ``viewer_….html`` and return ``http://127.0.0.1:…/3Dmol-min.js``.

    If download fails or ``PY3DMOL_NO_MIRROR`` is set, returns ``source_url`` unchanged.
    """
    if os.environ.get("PY3DMOL_NO_MIRROR", "").strip().lower() in ("1", "true", "yes"):
        return source_url
    dest = srv.directory / "3Dmol-min.js"
    lock = _lock_for_mirror_dir(srv.directory)
    with lock:
        try:
            if dest.is_file() and dest.stat().st_size >= _MIN_3DMOL_BYTES:
                return f"{srv.root_url}/3Dmol-min.js"
        except OSError:
            pass
        try:
            import urllib.error
            import urllib.request

            req = urllib.request.Request(
                source_url,
                headers={"User-Agent": "ProteinPredictor-GUI/1"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            if len(data) < _MIN_3DMOL_BYTES:
                raise ValueError(f"3Dmol-min.js too small ({len(data)} bytes)")
            dest.write_bytes(data)
        except Exception:
            return source_url
    return f"{srv.root_url}/3Dmol-min.js"


@dataclass(frozen=True)
class _StaticServer:
    """Loopback HTTP root used for ``3Dmol-min.js`` and ``viewer.html``."""

    directory: Path
    root_url: str


def _handler_class_for_directory(directory: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    d = str(directory)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=d, **kwargs)

        def guess_type(self, path: str) -> str:
            lp = path.lower()
            if lp.endswith(".js") or lp.endswith(".mjs"):
                return "application/javascript"
            return super().guess_type(path)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store, max-age=0, must-revalidate")
            super().end_headers()

    return Handler


def _get_static_server(cache_key: str, js_library_source: Path | None) -> _StaticServer:
    """
    One server per ``cache_key``. Optionally seed ``3Dmol-min.js`` from disk (``PY3DMOL_JS_FILE``).
    Serves ``viewer.html`` written on each render (see :func:`_publish_viewer_iframe`).
    """
    with _servers_lock:
        hit = _servers.get(cache_key)
        if hit is not None:
            return hit
        d = Path(tempfile.mkdtemp(prefix="pp_static_"))
        if js_library_source is not None:
            shutil.copy2(js_library_source, d / "3Dmol-min.js")
        handler_cls = _handler_class_for_directory(d)
        httpd = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        port = int(httpd.server_address[1])
        threading.Thread(target=httpd.serve_forever, daemon=True, name=f"pp-static-{port}").start()
        root_url = f"http://127.0.0.1:{port}"
        srv = _StaticServer(directory=d, root_url=root_url)
        _servers[cache_key] = srv
        return srv


def _viewer_server_and_js_url() -> tuple[_StaticServer, str]:
    """Loopback server + the ``js=`` URL py3Dmol should use (prefer same-origin ``/3Dmol-min.js``)."""
    env_url = os.environ.get("PY3DMOL_JS_URL", "").strip()
    if env_url:
        srv = _get_static_server(f"url:{env_url}", None)
        return srv, _mirror_remote_3dmol_to_loopback(srv, env_url)
    fpath = _py3dmol_js_file_path()
    if fpath is not None:
        key = str(fpath.resolve())
        srv = _get_static_server(key, fpath)
        return srv, f"{srv.root_url}/3Dmol-min.js"
    srv = _get_static_server("__cdn_only__", None)
    return srv, _mirror_remote_3dmol_to_loopback(srv, _DEFAULT_3DMOL_CDN)


def _publish_viewer_iframe(
    inner_html: str,
    *,
    width: int,
    height: int,
    slot: str,
) -> None:
    """Write a named HTML page on the loopback server and embed it via ``src=`` (not ``srcdoc``)."""
    srv, _ = _viewer_server_and_js_url()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in slot)[:48] or "main"
    fname = f"viewer_{safe}.html"
    doc = (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
        "<title>Structure viewer</title>\n"
        f"{_VIEWER_PAGE_STYLE}\n"
        "</head>\n"
        f"<body><div id=\"pp-outer\">{inner_html}</div></body>\n</html>\n"
    )
    (srv.directory / fname).write_text(doc, encoding="utf-8")
    import streamlit.components.v1 as components

    bust = time.monotonic_ns()
    components.iframe(
        f"{srv.root_url}/{fname}?v={bust}",
        width=width,
        height=height + 24,
        scrolling=True,
    )


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

    - ``PY3DMOL_JS_URL`` — fetch from this URL; file is **mirrored** onto loopback when possible.
    - ``PY3DMOL_JS_FILE`` — copy is **served on 127.0.0.1** together with ``viewer.html``.
    - else default CDN URL, **mirrored** to loopback (same-origin script; set ``PY3DMOL_NO_MIRROR=1`` to skip).
    """
    return _viewer_server_and_js_url()[1]


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
        "(icons, UI). The viewer is a **nested iframe** whose **document** is `http://127.0.0.1:…/viewer_….html` — "
        "its requests (including **`3Dmol-min.js`**, now usually **same-origin** on that loopback) show under "
        "that frame’s context, not “top”.\n\n"
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
        st.markdown(
            f"[Open 3Dmol.js URL in new tab]({url}) — you should see minified JavaScript."
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
        out["3dmol_js_mode"] = "local file + viewer.html on 127.0.0.1 (iframe src=, not srcdoc)"
        out["PY3DMOL_JS_FILE_resolved"] = str(fpath.resolve())
        srv, js_u = _viewer_server_and_js_url()
        out["3dmol_effective_js_url"] = js_u
        out["viewer_loopback_root"] = srv.root_url
        out["3dmol_same_origin_script"] = "yes — PY3DMOL_JS_FILE is served as /3Dmol-min.js on loopback."
    else:
        out["3dmol_js_mode"] = "async URL (py3Dmol loadScriptAsync) + viewer page on loopback"
        srv, js_u = _viewer_server_and_js_url()
        out["3dmol_js_url"] = js_u
        out["viewer_loopback_root"] = srv.root_url
        if js_u.startswith(srv.root_url):
            out["3dmol_same_origin_script"] = (
                "yes — 3Dmol-min.js is served from the same loopback origin as the viewer "
                "(mirrored from CDN unless PY3DMOL_NO_MIRROR=1)."
            )
        else:
            out["3dmol_same_origin_script"] = (
                "no — script loads from a remote host (mirror failed or PY3DMOL_NO_MIRROR=1); "
                "adblock / tracking prevention in nested iframes may block it."
            )
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
        f"{_VIEWER_PAGE_STYLE}\n"
        "</head>\n"
        f'<body><div id="pp-outer">{inner}</div></body>\n</html>\n'
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
    _publish_viewer_iframe(inner, width=1000, height=height, slot=f"bg_{key_prefix}")


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
        _publish_viewer_iframe(html, width=920, height=height, slot=key_prefix)
        standalone = build_standalone_viewer_html(pdb_text, width=900, height=height)
        if standalone:
            st.download_button(
                label="Download standalone viewer (open in Chrome)",
                data=standalone.encode("utf-8"),
                file_name="structure_viewer_standalone.html",
                mime="text/html",
                key=f"{key_prefix}_standalone_html",
                help="Uses the same default 3Dmol.js CDN as the in-app viewer; open in Chrome if the embed is blank.",
            )
        if show_troubleshoot_caption:
            n_atom = sum(
                1
                for line in pdb_text.splitlines()
                if line.startswith("ATOM") or line.startswith("HETATM")
            )
            st.caption(
                f"Viewer fragment generated ({len(html)} chars, ~{n_atom} ATOM/HETATM lines), "
                "shown via **loopback iframe src=**. "
                "**DevTools:** py3Dmol uses **`viewer_<id>`**, not `viewer` — try "
                "`Object.keys(window).filter(k => k.startsWith('viewer_'))` in the **viewer page** console. "
                "If standalone download is also blank, inspect PDB text (pseudo-PDB must look like real ATOM lines)."
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
