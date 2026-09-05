#!/usr/bin/env bash
set -euo pipefail

while IFS= read -r source; do
  chapter_dir="${source%/main.typ}"
  output="build/${chapter_dir#chapters/}.pdf"
  mkdir -p "$(dirname "$output")"
  typst compile --root . "$source" "$output"
done < <(rg --files chapters -g 'main.typ' | sort)
