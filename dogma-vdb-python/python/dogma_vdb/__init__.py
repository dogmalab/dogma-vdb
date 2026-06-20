"""dogma-vdb — Portable vector database with hybrid search."""

from dogma_vdb._native import (
    Collection,
    Document,
    ScoredDocument,
    Metric,
)

__all__ = ["Collection", "Document", "ScoredDocument", "Metric"]
__version__ = "0.1.0"
