#!/usr/bin/env bash
# Build note pages with the cells run the way the live kernel runs them: cwd = workspace/, dlviz importable,
# all caches inside the project.   Usage: tools/notes/build/build_dl.sh <subject-dir-name> [page-stem ...]
#                                         tools/notes/build/build_dl.sh home        (the course home page)
set -e
R="$(cd "$(dirname "$0")/../../.." && pwd)"
export TMP="$R/tools/notes/.tmp" TEMP="$R/tools/notes/.tmp" MPLCONFIGDIR="$R/tools/notes/.tmp/mpl" \
       TORCH_HOME="$R/tools/notes/.tmp/torch" PYTHONIOENCODING=utf-8 PYTHONPATH="$R/workspace"
cd "$R/workspace"
if [ "$1" = home ]; then
  NOTES_OUT="$R/notes" ../.venv/Scripts/python.exe ../tools/notes/build/build.py "$R/tools/notes/src" "$R/notes"
else
  s="$1"; shift
  ../.venv/Scripts/python.exe ../tools/notes/build/build.py "$R/tools/notes/src/$s" "$R/notes" "$@"
fi
