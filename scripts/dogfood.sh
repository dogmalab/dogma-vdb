#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────
# dogfood.sh — Dogfooding automático para dogma-vdb
#
# Usa dogma-vdb-rag (el propio proyecto) para indexar y
# consultar el código fuente de dogma-vdb.
#
# Uso:
#   ./scripts/dogfood.sh [--watch] [--query "tu pregunta"]
#   ./scripts/dogfood.sh --bench          # benchmark rápido
# ─────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAG_BIN="${SCRIPT_DIR}/target/release/dogma-vdb-rag"
COLLECTION="${SCRIPT_DIR}/dogfood/dogma-vdb"
SOURCE="${SCRIPT_DIR}/src"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[dogfood]${NC} $1"; }
ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Build ──────────────────────────────────────────────
build_rag() {
    info "Compilando dogma-vdb-rag (release)..."
    cargo build --release -p dogma-vdb-rag 2>/dev/null
    if [ ! -f "$RAG_BIN" ]; then
        fail "No se encontró $RAG_BIN (compilación falló?)"
    fi
    ok "dogma-vdb-rag compilado"
}

# ── Ingest ─────────────────────────────────────────────
do_ingest() {
    info "Indexando $SOURCE → ${COLLECTION}.vdb"
    mkdir -p "$(dirname "$COLLECTION")"
    $RAG_BIN ingest \
        --source "$SOURCE" \
        --collection "$COLLECTION" \
        --extensions "rs,toml,md" \
        --hash --dim 64 \
        --chunker code
    if [ -f "${COLLECTION}.vdb" ]; then
        local size
        size=$(du -h "${COLLECTION}.vdb" | cut -f1)
        ok "Colección creada: ${COLLECTION}.vdb (${size})"
    else
        fail "No se creó la colección"
    fi
}

# ── Info ───────────────────────────────────────────────
do_info() {
    info "Metadata de la colección:"
    $RAG_BIN info --collection "$COLLECTION"
}

# ── Query ──────────────────────────────────────────────
do_query() {
    local query="$1"
    info "Consultando: ${query}"
    echo ""
    $RAG_BIN query \
        --collection "$COLLECTION" \
        --query "$query" \
        --top-k 3
}

# ── Hybrid query ───────────────────────────────────────
do_hybrid() {
    local query="$1"
    info "Consulta híbrida (vector + BM25 + RRF): ${query}"
    echo ""
    $RAG_BIN query \
        --collection "$COLLECTION" \
        --query "$query" \
        --top-k 3 \
        --hybrid
}

# ── Watch ──────────────────────────────────────────────
do_watch() {
    info "Vigilando cambios en $SOURCE (Ctrl+C para salir)..."
    info "La colección se actualizará automáticamente al modificar archivos"
    echo ""
    $RAG_BIN watch \
        --source "$SOURCE" \
        --collection "$COLLECTION" \
        --extensions "rs,toml,md" \
        --hash --dim 64 \
        --chunker code \
        --debounce 2
}

# ── Benchmark (inline) ─────────────────────────────────
do_bench() {
    info "Benchmark rápido de dogma-vdb-rag:"
    echo ""

    # 1) Ingest
    local start end elapsed
    start=$(date +%s%N)
    $RAG_BIN ingest \
        --source "$SOURCE" \
        --collection "$COLLECTION" \
        --extensions "rs,toml,md" \
        --hash --dim 64 \
        --chunker code 2>/dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    ok "Ingest: ${elapsed}ms"

    # 2) Info
    local doc_count
    doc_count=$($RAG_BIN info --collection "$COLLECTION" 2>/dev/null | grep -oP '\d+ docs|chunks:\s*\K\d+')
    ok "Documentos indexados: ${doc_count:-?}"

    # 3) Query x3
    for q in "Bm25Index" "MmapBackedStorage" "IvfPqIndex"; do
        start=$(date +%s%N)
        $RAG_BIN query \
            --collection "$COLLECTION" \
            --query "$q" \
            --top-k 3 2>/dev/null
        end=$(date +%s%N)
        elapsed=$(( (end - start) / 1000000 ))
        ok "Query '${q}': ${elapsed}ms"
    done

    # 4) Hybrid query
    start=$(date +%s%N)
    $RAG_BIN query \
        --collection "$COLLECTION" \
        --query "memory guard auto-detection" \
        --top-k 3 --hybrid 2>/dev/null
    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    ok "Hybrid query: ${elapsed}ms"

    # 5) Stats
    local total_size
    total_size=$(du -sh "${COLLECTION}.vdb" 2>/dev/null | cut -f1)
    if [ -f "${COLLECTION}.bm25" ]; then
        local bm25_size
        bm25_size=$(du -sh "${COLLECTION}.bm25" 2>/dev/null | cut -f1)
        echo ""
        info "Cache BM25: ${bm25_size} (ya construido, queries híbridas más rápidas)"
    fi
    echo ""
    info "Benchmark completo — colección: ${total_size:-?}"
}

# ── Clean ──────────────────────────────────────────────
do_clean() {
    rm -f "${COLLECTION}.vdb" "${COLLECTION}.bm25"
    ok "Colección limpiada"
}

# ── Main ───────────────────────────────────────────────
main() {
    cd "$SCRIPT_DIR"

    case "${1:-}" in
        --bench)
            build_rag
            do_ingest
            do_bench
            ;;
        --watch)
            build_rag
            do_ingest
            do_watch
            ;;
        --query)
            build_rag
            if [ ! -f "${COLLECTION}.vdb" ]; then
                do_ingest
            fi
            shift
            do_query "$*"
            ;;
        --hybrid)
            build_rag
            if [ ! -f "${COLLECTION}.vdb" ]; then
                do_ingest
            fi
            shift
            do_hybrid "$*"
            ;;
        --info)
            build_rag
            if [ ! -f "${COLLECTION}.vdb" ]; then
                fail "Ejecuta primero sin flags para crear la colección"
            fi
            do_info
            ;;
        --clean)
            do_clean
            ;;
        --help|-h)
            echo "Uso: $0 [--watch|--bench|--query <texto>|--hybrid <texto>|--info|--clean]"
            echo ""
            echo "Sin flags: ingesta completa + metadata + consultas de ejemplo"
            exit 0
            ;;
        *)
            build_rag
            do_ingest
            do_info
            echo ""
            do_query "Bm25Index"
            echo ""
            do_query "memory guard"
            echo ""
            do_hybrid "vector database"
            echo ""
            info "Usa --watch para mantenerlo actualizado automáticamente"
            info "Usa --bench para benchmark rápido"
            info "Usa --query <texto> para consulta única"
            info "Usa --clean para borrar la colección"
            ;;
    esac
}

main "$@"
