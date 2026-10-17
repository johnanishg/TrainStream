"""Optional live dashboard for the examples.

When the analytics project (``trainstream-analytics/`` next to this repository's
root, or the folder in ``TRAINSTREAM_ANALYTICS``) is present, the examples stream
every step to its browser UI, exactly like its own ``train.py``. Without it they
run unchanged and only print to the console. ``--no-ui`` turns it off.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DIRS = [os.environ.get("TRAINSTREAM_ANALYTICS"), os.path.join(HERE, "..", "trainstream-analytics")]


def add_args(ap) -> None:
    g = ap.add_argument_group("live dashboard")
    g.add_argument("--no-ui", action="store_true", help="do not stream to the analytics UI")
    g.add_argument("--ui-port", type=int, default=8800)
    g.add_argument("--no-browser", action="store_true", help="stream, but do not open a browser tab")
    g.add_argument("--exit-when-done", action="store_true", help="do not keep the UI server running afterwards")


def _module():
    for d in _DIRS:
        if d and os.path.isfile(os.path.join(d, "livereport.py")):
            d = os.path.abspath(d)
            if d not in sys.path:
                sys.path.insert(0, d)
            import livereport
            return livereport
    return None


class _Off:
    """Stand-in when the UI is unavailable or disabled: every call is a no-op."""

    rep = None

    def __getattr__(self, name):
        return lambda *a, **k: None

    def hf_callback(self, *a, **k):
        return None


def open_run(args, name: str):
    """A livereport.LiveRun for `name`, or a no-op object."""
    if getattr(args, "no_ui", False):
        return _Off()
    mod = _module()
    if mod is None:
        return _Off()
    try:
        return mod.LiveRun(name, port=args.ui_port, open_browser=not args.no_browser)
    except Exception as exc:  # port in use by something else, etc.
        print(f"[live] dashboard unavailable ({exc}); continuing without it")
        return _Off()


def start_server(args) -> bool:
    """Start (or find) the UI server in this process, for scripts whose
    training runs in child processes that then join it. True if available."""
    if getattr(args, "no_ui", False):
        return False
    mod = _module()
    if mod is None:
        return False
    import analytics
    server, _ = analytics.start_server(args.ui_port)
    if server is None and not analytics._existing_server(args.ui_port):
        print(f"[live] port {args.ui_port} is busy; continuing without the dashboard")
        return False
    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{args.ui_port}/")
    print(f"[live] dashboard: http://127.0.0.1:{args.ui_port}/")
    return True


def child_args(args) -> list:
    """Command-line flags that make a child process report to the same UI."""
    if getattr(args, "no_ui", False):
        return ["--no-ui"]
    return ["--ui-port", str(args.ui_port), "--no-browser", "--exit-when-done"]


def keep_serving(args, live=None) -> None:
    """Keep the dashboard up after training (Ctrl+C to quit), like train.py."""
    if getattr(args, "no_ui", False) or getattr(args, "exit_when_done", False):
        return
    if live is not None and getattr(live, "rep", None) is not None:
        live.serve_until_interrupted()
        return
    if _module() is not None:
        import time
        print(f"[live] done; dashboard still at http://127.0.0.1:{args.ui_port}/ (Ctrl+C to quit)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
