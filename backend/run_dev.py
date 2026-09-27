"""Dev entrypoint: uvicorn with reload that also watches .env files."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import uvicorn

_BACKEND_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _BACKEND_ROOT / ".env"
# uvicorn FileFilter is unreliable for dotfiles on Windows; touch this .py instead
_ENV_TRIGGER = _BACKEND_ROOT / "_env_reload_trigger.py"


def _env_watcher() -> None:
    """Poll .env mtime and touch a tracked .py file so WatchFiles always reloads."""
    last_mtime: float | None = None
    try:
        if _ENV_FILE.is_file():
            last_mtime = _ENV_FILE.stat().st_mtime
    except OSError:
        last_mtime = None

    if not _ENV_TRIGGER.is_file():
        _ENV_TRIGGER.write_text(
            "# Auto-generated: touched when .env changes so uvicorn reloads.\n",
            encoding="utf-8",
        )

    while True:
        time.sleep(0.75)
        try:
            if not _ENV_FILE.is_file():
                continue
            mtime = _ENV_FILE.stat().st_mtime
        except OSError:
            continue
        if last_mtime is None:
            last_mtime = mtime
            continue
        if mtime == last_mtime:
            continue
        last_mtime = mtime
        _ENV_TRIGGER.write_text(
            f"# Auto-generated: .env mtime={mtime}\n",
            encoding="utf-8",
        )
        print(f"[run_dev] .env changed -> reload trigger (mtime={mtime})", flush=True)


def main() -> None:
    # Thread lives in the reloader parent process (uvicorn.run reload=True).
    threading.Thread(target=_env_watcher, name="env-watcher", daemon=True).start()

    # ".*" in includes removes uvicorn's default exclude of all dotfiles.
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8092,
        reload=True,
        reload_includes=[".*", ".env", ".env.*", "*.env"],
        reload_dirs=["."],
    )


if __name__ == "__main__":
    main()
