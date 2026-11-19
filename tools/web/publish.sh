#!/usr/bin/env bash
# Publish the demo page to GitHub Pages: build both WebAssembly modules, lay
# the page out under its build's path (versioned.py) and push that as the
# only commit of the `gh-pages` branch. The build published before is kept
# beside the new one, so a page someone has open goes on finding its files
# until they reload; older builds are dropped. The page is built here, not on
# GitHub, because it includes nanocodec's encoder from the sibling checkout.
#
#   tools/web/publish.sh [--built]    # --built: publish web/pkg and web/nanocodec as they are
#
# Needs what build.sh and build-nanocodec.sh need.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
remote=$(git remote get-url origin)
revision=$(git rev-parse --short HEAD)
if [[ "${1:-}" != "--built" ]]; then
  tools/web/build.sh
  tools/web/build-nanocodec.sh
fi
site=$(mktemp -d)
trap 'rm -rf "$site"' EXIT
python3 tools/web/versioned.py "$site/page"
# The build the published index names, if there is one, comes along.
if git clone --quiet --depth 1 --branch gh-pages "$remote" "$site/published" 2>/dev/null; then
  previous=$(grep -o 'v/[0-9a-f]*' "$site/published/current.json" 2>/dev/null | head -n 1 || true)
  if [[ -n "$previous" && -d "$site/published/$previous" && ! -e "$site/page/$previous" ]]; then
    cp -r "$site/published/$previous" "$site/page/$previous"
  fi
fi
cd "$site/page"
# Without this GitHub runs the files through Jekyll, which drops some of them.
touch .nojekyll
git init --quiet --initial-branch=gh-pages
git add --all
git -c user.name="$(git -C "$OLDPWD" config user.name)" -c user.email="$(git -C "$OLDPWD" config user.email)" \
  commit --quiet --message "The demo page, built from $revision"
git push --force "$remote" gh-pages
