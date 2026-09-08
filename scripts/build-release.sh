#!/usr/bin/env bash
set -euo pipefail

typst compile --root . --input handbook=true main.typ build/the-foundation-model-handbook-v1.0.pdf
