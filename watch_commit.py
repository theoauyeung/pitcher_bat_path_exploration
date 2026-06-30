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

POLL_INTERVAL = 2    # seconds between git status checks
DEBOUNCE_SECS = 30   # seconds of quiet before committing — long enough for plot generation

ROOT = Path(__file__).parent


def _git(*args):
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=ROOT,
    )


def _has_changes():
    return bool(_git("status", "--porcelain").stdout.strip())


def _pull_if_behind():
    """Fetch origin and rebase if remote has commits we don't have locally.

    Stashes any uncommitted local changes first, rebases onto the remote branch,
    then pops the stash so local edits survive. Returns True on success.
    """
    _git("fetch", "origin")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    count_result = _git("rev-list", "--count", f"HEAD..origin/{branch}")
    if count_result.returncode != 0 or count_result.stdout.strip() in ("", "0"):
        return True  # remote ref doesn't exist yet or already up to date

    n = count_result.stdout.strip()
    print(f"  remote has {n} new commit(s) on {branch}, pulling first...")

    # Stash local changes (including untracked) so rebase has a clean tree
    stash = _git("stash", "push", "-u", "-m", "watch_commit: pre-pull")
    stashed = "Saved" in stash.stdout

    rebase = _git("rebase", f"origin/{branch}")
    if rebase.returncode != 0:
        print(f"  rebase failed: {rebase.stderr.strip()}", file=sys.stderr)
        _git("rebase", "--abort")
        if stashed:
            _git("stash", "pop")
        return False

    if stashed:
        pop = _git("stash", "pop")
        if pop.returncode != 0:
            print(f"  stash pop conflict after pull: {pop.stderr.strip()}", file=sys.stderr)
            return False

    print(f"  pulled {n} remote commit(s) successfully.")
    return True


def _commit():
    if not _pull_if_behind():
        return

    _git("add", ".")
    staged = _git("diff", "--cached", "--name-only").stdout.strip().splitlines()
    if not staged:
        return

    label = ", ".join(staged[:3])
    if len(staged) > 3:
        label += f" (+{len(staged) - 3} more)"
    msg = f"auto: {label}"

    result = _git("commit", "-m", msg)
    if result.returncode != 0:
        print(f"  commit failed: {result.stderr.strip()}", file=sys.stderr)
        return

    short = _git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"  [{short}] {msg} — pushing...", end="", flush=True)
    push = _git("push", "origin", "HEAD")
    if push.returncode == 0:
        print(" done.")
    else:
        print(f" push failed: {push.stderr.strip()}", file=sys.stderr)


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
