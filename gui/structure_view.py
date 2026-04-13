"""
In-browser protein structure (py3Dmol) and optional PyMOL script download for the Streamlit **Structure** tab.

**3Dmol.js loading:** py3Dmol injects a ``<script>`` for the library. By default the app **downloads**
the chosen URL once into a local static dir and serves ``/3Dmol-min.js`` from **the same origin**
as ``viewer_….html`` (avoids third‑party / tracking‑prevention blocks on cross‑origin scripts inside
nested iframes). The iframe and script URLs use the browser’s **Host** (or **X-Forwarded-Host**)
when available so LAN / tunnel access works; otherwise **127.0.0.1**. Override behavior:

- ``PY3DMOL_NO_MIRROR=1`` — pass the remote URL straight to py3Dmol (old behavior).
- ``PY3DMOL_JS_URL`` / ``PY3DMOL_JS_FILE`` — same mirror logic unless ``NO_MIRROR``.

Viewer HTML is served over a local ``HTTPServer`` (listens on **0.0.0.0**) and embedded with ``iframe``
``src=`` (not ``srcdoc``) so large PDBs and script loading behave reliably in nested frames.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

_logger = logging.getLogger(__name__)

# Last mirror failure (any server dir); cleared on successful download/write.
_last_mirror_error: str | None = None

# Official Pitt-hosted build (avoids blocked/misrouted jsDelivr in some networks).
# Override with PY3DMOL_JS_URL, e.g. https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js for a pinned mirror.
_DEFAULT_3DMOL_CDN = "https://3dmol.csb.pitt.edu/build/3Dmol-min.js"
_DEFAULT_3DMOL_CDN_ALTERNATES = (
    "https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js",
    "https://unpkg.com/3dmol@2.5.4/build/3Dmol-min.js",
)

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


def _forwarded_or_host_header() -> str | None:
    """First hop of X-Forwarded-Host or Host from the current Streamlit request, if any."""
    try:
        headers = getattr(st.context, "headers", None)
    except (AttributeError, RuntimeError):
        return None
    if headers is None:
        return None
    for key in ("X-Forwarded-Host", "x-forwarded-host", "Host", "host"):
        if hasattr(headers, "get"):
            raw = headers.get(key)
        else:
            raw = None
        if raw is None and isinstance(headers, dict):
            raw = headers.get(key)
        if not raw:
            continue
        s = str(raw).strip()
        if s:
            return s.split(",")[0].strip()
    return None


def _host_header_to_hostname(host: str) -> str:
    """
    Strip default port suffix from Host-style values for use in http://HOST:static_port URLs.

    Keeps bracketed IPv6 literals (``[::1]``) intact.
    """
    host = host.strip()
    if not host:
        return "127.0.0.1"
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            return host[: end + 1]
        return host
    if host.count(":") == 1:
        name, port = host.rsplit(":", 1)
        if port.isdigit():
            return name
    return host


def _browser_static_host() -> str:
    """Hostname the **browser** should use to reach the static HTTP server."""
    raw = _forwarded_or_host_header()
    if raw:
        return _host_header_to_hostname(raw)
    return "127.0.0.1"


def _lock_for_mirror_dir(directory: Path) -> threading.Lock:
    key = str(directory.resolve())
    with _servers_lock:
        if key not in _mirror_locks:
            _mirror_locks[key] = threading.Lock()
        return _mirror_locks[key]


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _candidate_mirror_urls(source_url: str) -> list[str]:
    """
    Candidate URLs to download for local mirroring.

    Resolution strategy:
    - first, the requested source URL
    - if using the default CDN URL, try known alternates
    - if ``PY3DMOL_ALLOW_INSECURE_HTTP_FALLBACK`` is truthy (default: true), also
      try ``http://`` variants because some local Python SSL stacks fail handshakes.
    """
    urls: list[str] = [source_url]
    if source_url == _DEFAULT_3DMOL_CDN:
        urls.extend(_DEFAULT_3DMOL_CDN_ALTERNATES)
    if _truthy_env("PY3DMOL_ALLOW_INSECURE_HTTP_FALLBACK", default=True):
        https_urls = [u for u in urls if u.startswith("https://")]
        urls.extend("http://" + u[len("https://") :] for u in https_urls)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def _download_js_bytes(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "petase-thermostability-benchmark-GUI/1"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _mirror_remote_3dmol_to_loopback(srv: _StaticServer, source_url: str) -> str:
    """
    Write ``3Dmol-min.js`` next to ``viewer_….html`` and return same-origin URL under
    :meth:`_StaticServer.browser_base_url`.

    If download fails or ``PY3DMOL_NO_MIRROR`` is set, returns ``source_url`` unchanged.
    """
    global _last_mirror_error

    if os.environ.get("PY3DMOL_NO_MIRROR", "").strip().lower() in ("1", "true", "yes"):
        return source_url
    dest = srv.directory / "3Dmol-min.js"
    lock = _lock_for_mirror_dir(srv.directory)
    with lock:
        try:
            if dest.is_file() and dest.stat().st_size >= _MIN_3DMOL_BYTES:
                _last_mirror_error = None
                return f"{srv.browser_base_url()}/3Dmol-min.js"
        except OSError:
            pass
        errs: list[str] = []
        for candidate_url in _candidate_mirror_urls(source_url):
            try:
                data = _download_js_bytes(candidate_url)
                if len(data) < _MIN_3DMOL_BYTES:
                    raise ValueError(f"3Dmol-min.js too small ({len(data)} bytes)")
                dest.write_bytes(data)
                _last_mirror_error = None
                if candidate_url != source_url:
                    _logger.info("3Dmol mirror fallback succeeded: %s", candidate_url)
                return f"{srv.browser_base_url()}/3Dmol-min.js"
            except Exception as e:  # noqa: BLE001
                errs.append(f"{candidate_url} -> {type(e).__name__}: {e}")
        _last_mirror_error = " ; ".join(errs[:3])
        if len(errs) > 3:
            _last_mirror_error += f" ; ... ({len(errs)} total attempts)"
        _logger.warning("3Dmol mirror failed for all candidates: %s", _last_mirror_error)
        return source_url
    return f"{srv.browser_base_url()}/3Dmol-min.js"


@dataclass(frozen=True)
class _StaticServer:
    """Local HTTP server for ``3Dmol-min.js`` and ``viewer.html`` (browser uses :meth:`browser_base_url`)."""

    directory: Path
    port: int

    def browser_base_url(self) -> str:
        h = _browser_static_host()
        if h.startswith("["):
            return f"http://{h}:{self.port}"
        return f"http://{h}:{self.port}"


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
        httpd = http.server.HTTPServer(("0.0.0.0", 0), handler_cls)
        port = int(httpd.server_address[1])
        threading.Thread(target=httpd.serve_forever, daemon=True, name=f"pp-static-{port}").start()
        srv = _StaticServer(directory=d, port=port)
        _servers[cache_key] = srv
        return srv


def _viewer_server_and_js_url() -> tuple[_StaticServer, str]:
    """Local static server + the ``js=`` URL py3Dmol should use (prefer same-origin ``/3Dmol-min.js``)."""
    env_url = os.environ.get("PY3DMOL_JS_URL", "").strip()
    if env_url:
        srv = _get_static_server(f"url:{env_url}", None)
        return srv, _mirror_remote_3dmol_to_loopback(srv, env_url)
    fpath = _py3dmol_js_file_path()
    if fpath is not None:
        key = str(fpath.resolve())
        srv = _get_static_server(key, fpath)
        return srv, f"{srv.browser_base_url()}/3Dmol-min.js"
    srv = _get_static_server("__cdn_only__", None)
    return srv, _mirror_remote_3dmol_to_loopback(srv, _DEFAULT_3DMOL_CDN)


def _publish_viewer_iframe(
    inner_html: str,
    *,
    width: int,
    height: int,
    slot: str,
) -> None:
    """Write a named HTML page on the static server and embed it via ``src=`` (not ``srcdoc``)."""
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
        f"{srv.browser_base_url()}/{fname}?v={bust}",
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

    - ``PY3DMOL_JS_URL`` — fetch from this URL; file is **mirrored** onto the static server when possible.
    - ``PY3DMOL_JS_FILE`` — copy is **served** from the same origin as ``viewer_….html`` (browser host from Streamlit headers).
    - else default CDN URL, **mirrored** to that origin (same-origin script; set ``PY3DMOL_NO_MIRROR=1`` to skip).
    """
    return _viewer_server_and_js_url()[1]


def effective_3dmol_js_url() -> str:
    """Resolved URL passed to py3Dmol (starts static server if ``PY3DMOL_JS_FILE`` is set)."""
    return _py3dmol_js_url_for_view()


def probe_3dmol_js_url(url: str, *, read_bytes: int = 256) -> tuple[bool, str]:
    """HTTP GET from the Streamlit server process (same machine as the static 3Dmol server)."""
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "petase-thermostability-benchmark-GUI/1"},
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
    srv, _ = _viewer_server_and_js_url()
    viewer_origin = srv.browser_base_url()
    st.markdown(
        "**Why Network only shows `image/svg+xml`:** that list is usually for the **main Streamlit page** "
        "(icons, UI). The viewer is a **nested iframe** whose **document** is served from the static viewer "
        f"server (typically `{viewer_origin}/viewer_….html`) — "
        "its requests (including **`3Dmol-min.js`**, usually **same-origin** there) show under "
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
        out["3dmol_js_mode"] = "local file + viewer.html on static server (iframe src=, not srcdoc)"
        out["PY3DMOL_JS_FILE_resolved"] = str(fpath.resolve())
        srv, js_u = _viewer_server_and_js_url()
        out["3dmol_effective_js_url"] = js_u
        base = srv.browser_base_url()
        out["viewer_static_base"] = base
        out["viewer_browser_host"] = _browser_static_host()
        out["streamlit_host_header"] = _forwarded_or_host_header() or "(not available in this context)"
        out["3dmol_same_origin_script"] = "yes — PY3DMOL_JS_FILE is served as /3Dmol-min.js on the viewer origin."
    else:
        out["3dmol_js_mode"] = "async URL (py3Dmol loadScriptAsync) + viewer page on static server"
        srv, js_u = _viewer_server_and_js_url()
        out["3dmol_js_url"] = js_u
        base = srv.browser_base_url()
        out["viewer_static_base"] = base
        out["viewer_browser_host"] = _browser_static_host()
        out["streamlit_host_header"] = _forwarded_or_host_header() or "(not available in this context)"
        if js_u.startswith(base):
            out["3dmol_same_origin_script"] = (
                "yes — 3Dmol-min.js is served from the same origin as the viewer "
                "(mirrored from CDN unless PY3DMOL_NO_MIRROR=1)."
            )
        else:
            out["3dmol_same_origin_script"] = (
                "no — script loads from a remote host (mirror failed or PY3DMOL_NO_MIRROR=1); "
                "adblock / tracking prevention in nested iframes may block it."
            )
    if _last_mirror_error:
        out["3dmol_mirror_last_error"] = _last_mirror_error
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
    Full HTML document using a direct 3Dmol.js bootstrap (no py3Dmol dependency).
    """
    js = js_url or _DEFAULT_3DMOL_CDN
    inner = _direct_3dmol_fragment(
        pdb_text,
        width=width,
        height=height,
        style="cartoon_amino",
        spin=False,
        js_url=js,
        slot="standalone",
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
        "<title>petase-thermostability-benchmark — structure viewer</title>\n"
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


def _direct_style_js(*, style: str, has_backbone: bool) -> str:
    """JavaScript snippet that applies the requested style to ``v``."""
    if style.startswith("cartoon") and not has_backbone:
        return (
            "v.setStyle({atom:'CA'},{sphere:{radius:0.45,colorscheme:'amino'}});"
            "v.addStyle({atom:'CA'},{line:{linewidth:2.0,colorscheme:'amino'}});"
        )
    if style == "surface":
        # Avoid WebGL surface path (can render blank on some stacks); emulate a
        # bulky surface-like look using dense spheres plus a faint backbone trace.
        return (
            "v.setStyle({},{sphere:{radius:0.65,colorscheme:'amino'}});"
            "v.addStyle({hetflag:false},{stick:{radius:0.10,color:'lightgrey'}});"
        )
    if style == "cartoon_sticks":
        return (
            # Reliable across browsers/GPU stacks where cartoon can be invisible.
            "v.addStyle({hetflag:false},{stick:{radius:0.18,color:'hotpink'}});"
            "v.addStyle({atom:'CA'},{sphere:{radius:0.28,color:'white'}});"
        )
    if style == "cartoon_chain":
        return (
            # Render a chain-like silhouette using robust atom primitives.
            "v.setStyle({},{line:{linewidth:2.2,color:'white'}});"
            "v.addStyle({hetflag:false},{stick:{radius:0.16,color:'deepskyblue'}});"
        )
    return (
        "v.setStyle({},{stick:{radius:0.16,colorscheme:'amino'}});"
        "v.addStyle({atom:'CA'},{sphere:{radius:0.25,color:'white'}});"
    )


def _direct_3dmol_fragment(
    pdb_text: str,
    *,
    width: int,
    height: int,
    style: str,
    spin: bool,
    js_url: str,
    slot: str,
) -> str:
    """
    Self-contained 3Dmol.js bootstrap fragment.

    We avoid py3Dmol's generated HTML here because some browser/runtime combinations
    render blank despite successful script delivery.
    """
    safe_slot = "".join(c if c.isalnum() or c in "-_" else "_" for c in slot)[:48] or "viewer"
    token = f"{safe_slot}_{time.monotonic_ns()}"
    div_id = f"pp3dmol_{token}"
    viewer_name = f"viewer_{token}"
    style_js = _direct_style_js(style=style, has_backbone=_has_backbone_atoms(pdb_text))
    spin_js = "v.spin(true);" if spin else ""
    err_css = "color:#ffb4b4;background:#220;padding:8px;border:1px solid #733;border-radius:6px;font:12px/1.4 monospace;"
    return (
        f'<div id="{div_id}" style="width:{width}px;height:{height}px;max-width:100%;position:relative;"></div>\n'
        "<script>\n"
        "(function(){\n"
        f"  const div = document.getElementById({json.dumps(div_id)});\n"
        "  if (!div) return;\n"
        f"  const src = {json.dumps(js_url)};\n"
        "  function fail(msg, err){\n"
        f"    div.innerHTML = '<pre style=\"{err_css}\">'+msg+'</pre>';\n"
        "    if (err) { console.error('PP3DMOL', msg, err); }\n"
        "  }\n"
        "  function init(){\n"
        "    if (!window.$3Dmol) { fail('3Dmol.js loaded but window.$3Dmol is missing'); return; }\n"
        "    try {\n"
        "      const v = window.$3Dmol.createViewer(div, {backgroundColor:'#0e1117'});\n"
        f"      v.addModel({json.dumps(pdb_text)}, 'pdb');\n"
        f"      {style_js}\n"
        f"      {spin_js}\n"
        "      v.zoomTo();\n"
        "      v.render();\n"
        f"      window[{json.dumps(viewer_name)}] = v;\n"
        "      div.setAttribute('data-pp3dmol', 'ok');\n"
        "    } catch (err) {\n"
        "      fail('3Dmol render failed: ' + String(err), err);\n"
        "    }\n"
        "  }\n"
        "  if (window.$3Dmol) { init(); return; }\n"
        "  const prior = Array.from(document.scripts).find((s) => s.src === src);\n"
        "  if (prior) {\n"
        "    prior.addEventListener('load', init, {once:true});\n"
        "    prior.addEventListener('error', () => fail('3Dmol.js failed to load from ' + src), {once:true});\n"
        "    setTimeout(() => { if (!window.$3Dmol) fail('3Dmol.js load timeout from ' + src); }, 8000);\n"
        "    return;\n"
        "  }\n"
        "  const s = document.createElement('script');\n"
        "  s.src = src;\n"
        "  s.async = true;\n"
        "  s.onload = init;\n"
        "  s.onerror = () => fail('3Dmol.js failed to load from ' + src);\n"
        "  document.head.appendChild(s);\n"
        "  setTimeout(() => { if (!window.$3Dmol) fail('3Dmol.js load timeout from ' + src); }, 8000);\n"
        "})();\n"
        "</script>\n"
    )


def _has_backbone_atoms(pdb_text: str) -> bool:
    """True when N/CA/C backbone atoms are present in ATOM/HETATM records."""
    names: set[str] = set()
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
        if atom_name:
            names.add(atom_name)
            if {"N", "CA", "C"}.issubset(names):
                return True
    return False


def _apply_ca_trace_fallback_style(view: object) -> None:
    """Fallback style for pseudo/trace-only models that cannot render as cartoon."""
    view.setStyle({"atom": "CA"}, {"sphere": {"radius": 0.45, "colorscheme": "amino"}})
    view.addStyle({"atom": "CA"}, {"line": {"linewidth": 2.0, "colorscheme": "amino"}})


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
    js = _py3dmol_js_url_for_view()
    inner = _direct_3dmol_fragment(
        pdb_text,
        width=980,
        height=height,
        style="cartoon_chain",
        spin=True,
        js_url=js,
        slot=f"bg_{key_prefix}",
    )
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
    """Show in-browser 3Dmol.js viewer; always offer PyMOL script download."""
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

    js = _py3dmol_js_url_for_view()
    html = _direct_3dmol_fragment(
        pdb_text,
        width=900,
        height=height,
        style=style,
        spin=spin,
        js_url=js,
        slot=key_prefix,
    )
    _publish_viewer_iframe(html, width=920, height=height, slot=key_prefix)
    standalone = build_standalone_viewer_html(pdb_text, width=900, height=height, js_url=js)
    if standalone:
        st.download_button(
            label="Download standalone viewer (open in Chrome)",
            data=standalone.encode("utf-8"),
            file_name="structure_viewer_standalone.html",
            mime="text/html",
            key=f"{key_prefix}_standalone_html",
            help="Uses direct 3Dmol.js bootstrap; open in Chrome if the embed is blank.",
        )
    if show_troubleshoot_caption:
        n_atom = sum(
            1
            for line in pdb_text.splitlines()
            if line.startswith("ATOM") or line.startswith("HETATM")
        )
        st.caption(
            f"Viewer fragment generated ({len(html)} chars, ~{n_atom} ATOM/HETATM lines), "
            "shown via **iframe src=** to the local static server (browser-reachable host). "
            "Viewer object is exported as **`viewer_<id>`** in the iframe page; "
            "if blank persists, the page now prints direct 3Dmol load/render errors inline."
        )

    st.download_button(
        label="Download PyMOL script (load_view.pml)",
        data=pymol_load_script("uploaded.pdb"),
        file_name="load_view.pml",
        mime="text/plain",
        key=f"{key_prefix}_pml",
        help="Save your PDB as uploaded.pdb in the same folder, or edit the load line inside the script.",
    )
