#!/bin/bash
# ============================================================================
# Compile MACA (MetaX) custom op bitcode libraries for dl.custom().
#
# Usage:
#   bash compile_maca_bc.sh            compile all src/maca_*.cc
#   bash compile_maca_bc.sh -f         force recompilation
#
# Compiler (first match wins):
#   1. $MACA_MCCLANG            explicit override
#   2. $MACA_PATH/mccloud/bin/mcclang   MetaX device clang
#   3. mcclang / clang from PATH
#
# Output: ../../language/deeplink/bitcode/bc/*.bc
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BC_DIR="$SCRIPT_DIR/../../language/deeplink/bitcode/bc"

FORCE=false
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# --- Locate a bitcode-capable compiler --------------------------------------
MCCLANG="${MACA_MCCLANG:-}"
if [ -z "$MCCLANG" ] && [ -n "${MACA_PATH:-}" ] && [ -x "$MACA_PATH/mccloud/bin/mcclang" ]; then
    MCCLANG="$MACA_PATH/mccloud/bin/mcclang"
fi
if [ -z "$MCCLANG" ]; then
    if command -v mcclang &> /dev/null; then
        MCCLANG="mcclang"
    elif command -v clang &> /dev/null; then
        MCCLANG="clang"
    else
        echo "ERROR: Neither mcclang nor clang found."
        echo "  Set MACA_PATH or MACA_MCCLANG, or add mcclang/clang to PATH."
        exit 1
    fi
fi
echo "Compiler: $MCCLANG"

# --- Compile -----------------------------------------------------------------
mkdir -p "$BC_DIR"

shopt -s nullglob
sources=("$SRC_DIR"/maca_*.cc)
if [ ${#sources[@]} -eq 0 ]; then
    echo "No src/maca_*.cc sources found."
    exit 0
fi

for CPP in "${sources[@]}"; do
    base_name="$(basename "${CPP%.cc}")"
    BC="$BC_DIR/${base_name}.bc"
    if [ -f "$BC" ] && [ "$FORCE" != "true" ]; then
        echo "Bitcode file $BC already exists, skipping (use -f to rebuild)."
        continue
    fi
    echo "Compiling $CPP -> $BC ..."
    "$MCCLANG" -O3 -emit-llvm -c "$CPP" -o "$BC"
    echo "  OK: $BC"
done