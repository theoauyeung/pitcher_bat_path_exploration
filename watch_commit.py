"""
watch_commit.py
Auto-commits changes to git when files are saved.

Usage:
    .venv\Scripts\python.exe watch_commit.py

Polls every POLL_INTERVAL seconds. Once changes are detected, waits
DEBOUNCE_SECS of quiet (no new changes) before committing. Stop with Ctrl+C.
"""

import subprocess
import time
import sys
from pathlib import Path

POLL_INTERVAL = 2   # seconds between git status checks
DEBOUNCE_SECS = 5   # seconds of quiet before committing

ROOT = Path(__file__).parent


def _git(*args):
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=ROOT,
    )


def _has_changes():
    return bool(_git("status", "--porcelain").stdout.strip())


def _commit():
    _git("add", ".")
    staged = _git("diff", "--cached", "--name-only").stdout.strip().splitlines()
    if not staged:
        return

    label = ", ".join(staged[:3])
    if len(staged) > 3:
        label += f" (+{len(staged) - 3} more)"
    msg = f"auto: {label}"

    result = _git("commit", "-m", msg)
    if result.returncode == 0:
        short = _git("rev-parse", "--short", "HEAD").stdout.strip()
        print(f"  [{short}] {msg}")
    else:
        print(f"  commit failed: {result.stderr.strip()}", file=sys.stderr)


def main():
    print(f"Watching for changes (poll={POLL_INTERVAL}s, debounce={DEBOUNCE_SECS}s) — Ctrl+C to stop.")
    last_change = None

    while True:
        try:
            if _has_changes():
                now = time.time()
                if last_change is None:
                    last_change = now
                    print("  changes detected, waiting for quiet...")
                elif now - last_change >= DEBOUNCE_SECS:
                    _commit()
                    last_change = None
            else:
                last_change = None

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
