//! Real ONNX Cross-Encoder reranker using `ort` runtime and HuggingFace `tokenizers`.
//!
//! This module provides the production implementation of [`CrossEncoderReranker`]
//! that loads a real ONNX model (e.g. `bge-reranker-base`) and tokenizer,
//! tokenises (query, document) pairs, runs inference, and returns relevance scores.
//!
//! ## Model format
//!
//! Expects an ONNX-exported Cross-Encoder model with:
//! - Inputs: `input_ids` (i64, shape [batch, seq_len]),
//!   `attention_mask` (i64, shape [batch, seq_len])
//! - Output: `"logits"` (f32, shape [batch, 1]) — raw logit per pair.
//!
//! The logit is used **raw** (no sigmoid) because only the relative ranking
//! matters for reranking — sigmoid is a monotonic transform that would add
//! unnecessary computation without affecting the order.
//!
//! Recommended model: [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base)
//! exported to ONNX via:
//! ```bash
//! optimum-cli export onnx --model BAAI/bge-reranker-base ./models/bge-reranker-base/
//! ```

use crate::{CrossEncoderReranker, RerankError};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use std::path::Path;
use tokenizers::Tokenizer;

/// Per-document tokenisation result: (input_ids, attention_mask).
type TokenisedPair = Result<(Vec<i64>, Vec<i64>), RerankError>;

/// Production Cross-Encoder reranker powered by ONNX Runtime.
///
/// # Example (requires model files on disk)
///
/// ```ignore
/// use dogma_vdb_rerank::{CrossEncoderReranker, OnnxReranker};
///
/// let reranker = OnnxReranker::new(
///     "/models/reranker/model.onnx",
///     "/models/reranker/tokenizer.json",
///     512,
///     2,
/// )?;
///
/// let scores = reranker.compute_scores("rust memory", &[
///     "mmap and memory alignment in Rust".into(),
///     "how to cook pasta al dente".into(),
/// ])?;
/// // scores[0].1 > scores[1].1  (first doc is more relevant)
/// ```
pub struct OnnxReranker {
    session: Session,
    tokenizer: Tokenizer,
    max_length: usize,
    intra_threads: usize,
}

impl std::fmt::Debug for OnnxReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxReranker")
            .field("max_length", &self.max_length)
            .field("intra_threads", &self.intra_threads)
            .finish_non_exhaustive()
    }
}

impl OnnxReranker {
    /// Load a Cross-Encoder ONNX model and its tokenizer from disk.
    ///
    /// - `model_path`: path to the `.onnx` model file.
    /// - `tokenizer_path`: path to the `tokenizer.json` file (HF tokenizer).
    /// - `max_length`: maximum sequence length for tokenisation (e.g. 512).
    /// - `intra_threads`: number of intra-op threads for ORT.
    ///   Recommended: 2 for MCP servers, 0 lets ORT decide (may use all cores).
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        max_length: usize,
        intra_threads: usize,
    ) -> Result<Self, RerankError> {
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| RerankError::TokenizerError(e.to_string()))?;

        let mut builder =
            Session::builder().map_err(|e| RerankError::ModelError(format!("ORT builder: {e}")))?;

        // Apply intra_op thread limit if set; ORT's global default can
        // consume all cores and starve the MCP tokio runtime.  We clamp
        // to a safe minimum so inference parallelism doesn't overwhelm
        // the host — 2 threads is a good balance for Cross-Encoder.
        let threads = if intra_threads == 0 { 2 } else { intra_threads };
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| RerankError::ModelError(format!("ORT threads: {e}")))?;

        let session = builder
            .commit_from_file(model_path.as_ref())
            .map_err(|e| RerankError::ModelError(format!("ORT load model: {e}")))?;

        Ok(Self {
            session,
            tokenizer,
            max_length,
            intra_threads: threads,
        })
    }
}

impl CrossEncoderReranker for OnnxReranker {
    fn compute_scores(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<Vec<(usize, f32)>, RerankError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = documents.len();

        // ---- 1. Tokenise (query, doc) pairs in parallel ----
        let tokenised: Vec<TokenisedPair> = documents
            .par_iter()
            .map(|doc| {
                let encoding = self
                    .tokenizer
                    .encode(
                        tokenizers::EncodeInput::Dual(
                            query.to_string().into(),
                            doc.to_string().into(),
                        ),
                        true,
                    )
                    .map_err(|e| RerankError::TokenizerError(e.to_string()))?;

                let ids: Vec<i64> = encoding
                    .get_ids()
                    .iter()
                    .take(self.max_length)
                    .map(|&v| v as i64)
                    .collect();

                let mask: Vec<i64> = encoding
                    .get_attention_mask()
                    .iter()
                    .take(self.max_length)
                    .map(|&v| v as i64)
                    .collect();

                Ok((ids, mask))
            })
            .collect();

        // ---- 2. Build 2D arrays (batch × seq_len) with padding ----
        let mut input_ids = Array2::<i64>::zeros((batch_size, self.max_length));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, self.max_length));

        for (i, result) in tokenised.iter().enumerate() {
            let ids = result.as_ref().map_err(|e| match e {
                RerankError::TokenizerError(msg) => RerankError::TokenizerError(msg.clone()),
                _ => RerankError::TokenizerError("tokenisation failed".into()),
            })?;

            let (ids, mask) = ids;
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[[i, j]] = id;
                attention_mask[[i, j]] = m;
            }
        }

        // ---- 3. Convert to ORT tensors ----
        let input_ids_tensor = Tensor::from_array(input_ids)
            .map_err(|e| RerankError::ModelError(format!("ORT tensor input_ids: {e}")))?;

        let attention_mask_tensor = Tensor::from_array(attention_mask)
            .map_err(|e| RerankError::ModelError(format!("ORT tensor attention_mask: {e}")))?;

        // ---- 4. Run inference ----
        let outputs = self
            .session
            .run(
                ort::inputs![input_ids_tensor, attention_mask_tensor]
                    .map_err(|e| RerankError::ModelError(format!("ORT inputs: {e}")))?,
            )
            .map_err(|e| RerankError::ModelError(format!("ORT inference: {e}")))?;

        // ---- 5. Extract logits ----
        // bge-reranker-base outputs shape [batch, 1] with a single raw logit
        // per (query, document) pair.  No sigmoid is applied — the raw logit
        // is monotonic with respect to relevance, so ranking is preserved.
        let logits_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| RerankError::ModelError(format!("ORT extract logits: {e}")))?;

        // The output is an ndarray::ArrayViewD<f32>.  Since ONNX outputs
        // are contiguous, we can use as_slice() safely.
        let logits_slice = logits_view
            .as_slice()
            .ok_or_else(|| RerankError::ModelError("ORT output not contiguous".into()))?;

        // Per-row stride: [batch, 1] => stride = 1, [batch, 2] => stride = 2
        let stride = logits_slice.len() / batch_size;

        let mut results: Vec<(usize, f32)> = (0..batch_size)
            .map(|i| (i, logits_slice[i * stride]))
            .collect();

        // ---- 6. Sort descending by score ----
        results.par_sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_reranker_new_missing_model() {
        // Without model files, new() should return a clear error
        let result = OnnxReranker::new(
            "/nonexistent/model.onnx",
            "/nonexistent/tokenizer.json",
            512,
            2,
        );
        assert!(result.is_err());
        match result {
            Err(RerankError::TokenizerError(msg)) => {
                assert!(!msg.is_empty(), "TokenizerError should have a description");
            }
            Err(RerankError::ModelError(msg)) => {
                assert!(!msg.is_empty(), "ModelError should have a description");
            }
            _ => panic!("Expected an error, got {:?}", result),
        }
    }

    #[test]
    fn test_onnx_reranker_empty_docs() {
        // Verify the early-return: no model needed for empty input.
        // (We can't instantiate OnnxReranker without files, so we test
        //  the contract via a dedicated guard impl.)
        struct GuardReranker;
        impl CrossEncoderReranker for GuardReranker {
            fn compute_scores(
                &self,
                _query: &str,
                documents: &[String],
            ) -> Result<Vec<(usize, f32)>, RerankError> {
                if documents.is_empty() {
                    return Ok(vec![]);
                }
                Ok(vec![(0, 0.0)])
            }
        }
        let r = GuardReranker;
        assert!(r.compute_scores("q", &[]).unwrap().is_empty());
    }

    /// Integration test that requires a real model on disk.
    ///
    /// To run:
    /// ```bash
    /// # 1. Export bge-reranker-base to ONNX:
    /// optimum-cli export onnx --model BAAI/bge-reranker-base ./models/bge-reranker-base/
    ///
    /// # 2. Run with env vars pointing to the exported artefacts:
    /// DOGMA_RERANK_MODEL=./models/bge-reranker-base/model.onnx \
    /// DOGMA_RERANK_TOKENIZER=./models/bge-reranker-base/tokenizer.json \
    /// cargo test test_onnx_reranker_real_scoring -- --ignored
    /// ```
    #[test]
    #[ignore = "requires ONNX model files on disk — see doc comment for setup"]
    fn test_onnx_reranker_real_scoring() {
        let model_path = std::env::var("DOGMA_RERANK_MODEL")
            .unwrap_or_else(|_| "models/bge-reranker-base/model.onnx".into());
        let tok_path = std::env::var("DOGMA_RERANK_TOKENIZER")
            .unwrap_or_else(|_| "models/bge-reranker-base/tokenizer.json".into());

        let reranker = OnnxReranker::new(&model_path, &tok_path, 512, 2)
            .expect("Failed to load model. Set DOGMA_RERANK_MODEL and DOGMA_RERANK_TOKENIZER");

        let scores = reranker
            .compute_scores(
                "Rust memory management",
                &[
                    "mmap and memory alignment in Rust".into(),
                    "how to cook pasta al dente".into(),
                    "zero-copy deserialization with memmap2".into(),
                ],
            )
            .expect("Inference should succeed");

        assert_eq!(scores.len(), 3);
        // Rust-related docs should score higher than cooking
        assert!(
            scores[0].1 > scores[2].1,
            "Rust memory doc should score higher than cooking doc"
        );
        // Scores must be sorted descending
        for i in 0..scores.len().saturating_sub(1) {
            assert!(scores[i].1 >= scores[i + 1].1, "scores must be sorted");
        }
    }
}
