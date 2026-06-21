"""
deploy_github_pages.py — Build static site and push to GitHub Pages.

Runs build_static.py then commits and pushes docs/ to mark-richards/asl-hub.
Uses a fresh temp directory each run to avoid stale-state issues with
read-only git objects and Windows file locks on Drive-mounted paths.
"""

import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

log = logging.getLogger("deploy_github_pages")

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
GH_PAGES_BRANCH = "main"
_GH_REPO = "github.com/mark-richards/asl-hub.git"


def _remote_url() -> str:
    token = os.environ.get("GH_DEPLOY_TOKEN", "").strip()
    if token:
        return f"https://{token}@{_GH_REPO}"
    return f"https://{_GH_REPO}"


def _run(args: list, cwd: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # fail immediately instead of prompting
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(cwd), env=env)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _force_rmtree(path: Path) -> None:
    """Remove a directory tree, clearing read-only flags on Windows first."""
    def _on_error(func, fpath, _exc):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)
    shutil.rmtree(path, onerror=_on_error)


def build_static() -> bool:
    log.info("Building static site...")
    rc, out, err = _run([sys.executable, "build_static.py"], cwd=REPO_ROOT)
    if rc != 0:
        log.error("build_static.py failed (rc=%d): %s", rc, err or out)
        return False
    log.info("Static site built.")
    return True


def push_docs(round_num: int | None = None) -> bool:
    remote = _remote_url()
    tmp = Path(tempfile.mkdtemp(prefix="asl-hub-deploy-"))
    try:
        log.info("Cloning %s to %s...", _GH_REPO, tmp)
        rc, _, err = _run(["git", "clone", remote, str(tmp)], cwd=tmp.parent)
        if rc != 0:
            log.error("git clone failed: %s", err.replace(remote, f"https://{_GH_REPO}"))
            return False
        _run(["git", "config", "user.email", "deploy@sc-player-data"], cwd=tmp)
        _run(["git", "config", "user.name", "SC Deploy"], cwd=tmp)

        # Sync docs/ content into the clone (preserve .git)
        for item in tmp.iterdir():
            if item.name == ".git":
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        for item in DOCS_DIR.iterdir():
            if item.name == ".git":
                continue
            dest = tmp / item.name
            shutil.copytree(item, dest) if item.is_dir() else shutil.copy2(item, dest)

        label = f"Round {round_num}" if round_num else "update"
        msg = f"Deploy: {label} static site"

        _run(["git", "add", "-A"], cwd=tmp)
        rc, status_out, status_err = _run(["git", "status", "--porcelain"], cwd=tmp)
        if rc != 0:
            log.error("git status failed: %s", status_err)
            return False
        if not status_out:
            log.info("asl-hub unchanged — nothing to push.")
            return True

        rc, _, err = _run(["git", "commit", "-m", msg], cwd=tmp)
        if rc != 0:
            log.error("git commit failed: %s", err)
            return False

        log.info("Pushing to %s (%s)...", _GH_REPO, GH_PAGES_BRANCH)
        rc, _, err = _run(["git", "push", "origin", GH_PAGES_BRANCH], cwd=tmp)
        if rc != 0:
            log.error("git push failed: %s", err.replace(remote, f"https://{_GH_REPO}"))
            return False

        log.info("GitHub Pages updated: https://mark-richards.github.io/asl-hub/")
        return True
    finally:
        _force_rmtree(tmp)


def deploy(round_num: int | None = None) -> bool:
    if not build_static():
        return False
    return push_docs(round_num=round_num)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()
    ok = deploy(round_num=args.round)
    sys.exit(0 if ok else 1)
