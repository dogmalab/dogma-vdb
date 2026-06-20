// Minimal test: BruteForce only, no chunker, no files.
use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, Index};

fn main() {
    eprintln!("═══ Minimal BruteForce test ═══");
    eprintln!("Creating 100 random docs...");

    let docs: Vec<Document> = (0..100)
        .map(|i| {
            let emb: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32) / 100.0).collect();
            Document::builder(format!("d{i}"), format!("doc {i}"))
                .embedding(emb)
                .build()
        })
        .collect();

    eprintln!("Inserting into BruteForceIndex...");
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    bf.insert(&docs);
    eprintln!("Insert OK. {} docs", bf.len());

    eprintln!("Searching...");
    let query = vec![0.5f32; 64];
    let results = bf.search(&query, 5);
    eprintln!("Search OK. {} results", results.len());
    for r in &results {
        eprintln!("  {} score={}", r.document.id, r.score);
    }

    eprintln!("═══ Test OK ═══");
}
