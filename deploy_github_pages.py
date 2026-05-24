"""
deploy_github_pages.py — Build static site and push to GitHub Pages.

Runs build_static.py then commits and pushes docs/ to mark-richards/asl-hub.
Uses a temp directory outside Google Drive for git operations to avoid
filesystem permission issues with .git on Drive-mounted paths.
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("deploy_github_pages")

REPO_ROOT = Path(__file__).resolve().parent
DOCS_DIR = REPO_ROOT / "docs"
GH_PAGES_REMOTE = "https://github.com/mark-richards/asl-hub.git"
GH_PAGES_BRANCH = "main"
DEPLOY_CACHE = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))) / "asl-hub-deploy"


def _run(args: list, cwd: Path) -> tuple[int, str, str]:
    result = subprocess.run(args, capture_output=True, text=True, cwd=str(cwd))
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def build_static() -> bool:
    log.info("Building static site...")
    rc, out, err = _run([sys.executable, "build_static.py"], cwd=REPO_ROOT)
    if rc != 0:
        log.error("build_static.py failed (rc=%d): %s", rc, err or out)
        return False
    log.info("Static site built.")
    return True


def _ensure_deploy_clone() -> bool:
    """Clone asl-hub to DEPLOY_CACHE (or fetch if already cloned)."""
    git_dir = DEPLOY_CACHE / ".git"
    if git_dir.exists():
        log.info("Updating existing clone in %s...", DEPLOY_CACHE)
        rc, _, err = _run(["git", "fetch", "origin", GH_PAGES_BRANCH], cwd=DEPLOY_CACHE)
        if rc == 0:
            return True
        log.warning("git fetch failed (%s) — removing stale cache and re-cloning.", err)

    # Wipe everything (whether fetch failed or dir has no .git) and clone fresh
    shutil.rmtree(DEPLOY_CACHE, ignore_errors=True)
    log.info("Cloning %s to %s...", GH_PAGES_REMOTE, DEPLOY_CACHE)
    rc, _, err = _run(
        ["git", "clone", GH_PAGES_REMOTE, str(DEPLOY_CACHE)],
        cwd=DEPLOY_CACHE.parent,
    )
    if rc != 0:
        log.error("git clone failed: %s", err)
        return False
    _run(["git", "config", "user.email", "deploy@sc-player-data"], cwd=DEPLOY_CACHE)
    _run(["git", "config", "user.name",  "SC Deploy"], cwd=DEPLOY_CACHE)
    return True


def push_docs(round_num: int | None = None) -> bool:
    if not _ensure_deploy_clone():
        return False

    # Sync docs/ content into the clone (preserve .git)
    for item in DEPLOY_CACHE.iterdir():
        if item.name == ".git":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
    for item in DOCS_DIR.iterdir():
        if item.name == ".git":
            continue
        dest = DEPLOY_CACHE / item.name
        shutil.copytree(item, dest) if item.is_dir() else shutil.copy2(item, dest)

    label = f"Round {round_num}" if round_num else "update"
    msg = f"Deploy: {label} static site"

    _run(["git", "add", "-A"], cwd=DEPLOY_CACHE)
    rc, status_out, status_err = _run(["git", "status", "--porcelain"], cwd=DEPLOY_CACHE)
    if rc != 0:
        log.error("git status failed (not a git repo?): %s", status_err)
        return False
    if not status_out:
        log.info("asl-hub unchanged — nothing to push.")
        return True

    rc, _, err = _run(["git", "commit", "-m", msg], cwd=DEPLOY_CACHE)
    if rc != 0:
        log.error("git commit failed: %s", err)
        return False

    log.info("Pushing to %s (%s)...", GH_PAGES_REMOTE, GH_PAGES_BRANCH)
    rc, _, err = _run(["git", "push", "origin", GH_PAGES_BRANCH], cwd=DEPLOY_CACHE)
    if rc != 0:
        log.error("git push failed: %s", err)
        return False

    log.info("GitHub Pages updated: https://mark-richards.github.io/asl-hub/")
    return True


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
