// Test mínimo: solo BruteForce, sin chunker, sin archivos.
use dogma_vdb::distance::Metric;
use dogma_vdb::doc::Document;
use dogma_vdb::index::{BruteForceIndex, Index};

fn main() {
    eprintln!("═══ Test mínimo BruteForce ═══");
    eprintln!("Creando 100 docs aleatorios...");

    let docs: Vec<Document> = (0..100)
        .map(|i| {
            let emb: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32) / 100.0).collect();
            Document::builder(format!("d{i}"), format!("doc {i}"))
                .embedding(emb)
                .build()
        })
        .collect();

    eprintln!("Insertando en BruteForceIndex...");
    let mut bf = BruteForceIndex::new(Metric::Cosine);
    bf.insert(&docs);
    eprintln!("Insert OK. {} docs", bf.len());

    eprintln!("Buscando...");
    let query = vec![0.5f32; 64];
    let results = bf.search(&query, 5);
    eprintln!("Search OK. {} resultados", results.len());
    for r in &results {
        eprintln!("  {} score={}", r.document.id, r.score);
    }

    eprintln!("═══ Test OK ═══");
}
