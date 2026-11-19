#!/usr/bin/env bash
# Publish the demo page to GitHub Pages: build both WebAssembly modules, lay
# the page out under its build's path (versioned.py) and push that as the
# only commit of the `gh-pages` branch, replacing whatever was there. The
# page is built here, not on GitHub, because it includes nanocodec's encoder
# from the sibling checkout. Needs what build.sh and build-nanocodec.sh need.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
remote=$(git remote get-url origin)
tools/web/build.sh
tools/web/build-nanocodec.sh
site=$(mktemp -d)
trap 'rm -rf "$site"' EXIT
python3 tools/web/versioned.py "$site/page"
cd "$site/page"
# Without this GitHub runs the files through Jekyll, which drops some of them.
touch .nojekyll
git init --quiet --initial-branch=gh-pages
git add --all
git commit --quiet --message "The demo page, built from $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push --force "$remote" gh-pages
echo "published: enable Pages for the gh-pages branch in the repository's settings if this is the first time"
