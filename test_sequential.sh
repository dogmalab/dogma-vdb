#!/usr/bin/env bash
# ============================================================================
# test_sequential.sh — Test secuencial de dogma-vdb con telemetría
#
# Uso:
#   ./test_sequential.sh               # test completo (200 archivos)
#   ./test_sequential.sh --quick       # test rápido (20 archivos)
#   ./test_sequential.sh --watch       # test completo + monitorización
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

QUICK=false
WATCH=false
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=true ;;
        --watch) WATCH=true ;;
    esac
done

# ── Config ──────────────────────────────────────────────────────────────
SAMPLE_FILES=200
if $QUICK; then
    SAMPLE_FILES=20
fi

OUTFILE="/tmp/dogma-test-$$.txt"

echo "═══════════════════════════════════════════════════"
echo "  dogma-vdb Test Secuencial"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Archivos: $SAMPLE_FILES"
echo "═══════════════════════════════════════════════════"

# ── Compilar ────────────────────────────────────────────────────────────
echo ""
echo "── [build] Compilando release..."
cargo build --release --example test_sequential 2>&1 | tail -1

# ── Ejecutar ────────────────────────────────────────────────────────────
echo ""
echo "── [run] Ejecutando test_sequential..."
echo "  Output: $OUTFILE"
echo ""

if $WATCH; then
    # Mostrar en terminal + log a archivo
    ./target/release/examples/test_sequential 2>&1 | tee "$OUTFILE"
    EXIT_CODE=${PIPESTATUS[0]}
else
    # Solo log a archivo
    ./target/release/examples/test_sequential > "$OUTFILE" 2>&1
    EXIT_CODE=$?
fi

echo ""
echo "── [result] ──"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ COMPLETADO — código $EXIT_CODE"
    grep "RSS final\|todo liberado\|completado\|sin crashes" "$OUTFILE" 2>/dev/null | tail -3
else
    echo "  ❌ FALLÓ — código $EXIT_CODE"
    echo ""
    echo "  Últimas líneas del output:"
    tail -10 "$OUTFILE"
    echo ""
    echo "  Señal: $(( EXIT_CODE - 128 )) (si aplica)"
fi
echo "═══════════════════════════════════════════════════"
