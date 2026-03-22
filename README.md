# memvid-rs

[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Rust implementation of [memvid](https://github.com/Olow304/memvid) — encode text documents as QR codes in video files for compact storage with BERT-powered semantic retrieval.

## What is memvid-rs?

memvid-rs transforms text into searchable video archives:

1. **Chunk** documents into segments
2. **Encode** each chunk as a QR code frame
3. **Compile** frames into a ProRes video (lossless, intra-frame)
4. **Index** chunks with BERT embeddings for semantic search
5. **Retrieve** relevant chunks by meaning, not just keywords

The video format provides a durable, compact archive. The BERT index provides instant semantic search over the contents.

## Key Features

**Performance**
- Metal GPU acceleration on Apple Silicon (M-series) via HuggingFace Candle
- CUDA support for NVIDIA GPUs
- HNSW vector indexing for sub-second search across large corpora
- Parallel encoding pipeline (QR generation, BERT embedding, video encoding)

**Video Encoding**
- **ProRes codec** (intra-frame, lossless) — replaced H.265 for reliable QR decode on every frame
- Configurable ProRes profiles: proxy, lt, standard, hq, 4444, xq
- Hardware-accelerated encoding/decoding via VideoToolbox on macOS
- FFmpeg 8.0 (linked via ffmpeg-next)

**Machine Learning**
- BERT sentence embeddings (384-dimensional) via HuggingFace Candle — pure Rust, no Python
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Configurable model via `MEMVID_MODEL_NAME` env or TOML config
- Remote embedding API support (any OpenAI-compatible `/v1/embeddings` endpoint)
- Asymmetric query/document prefixes for instruction-tuned models

**Search**
- HNSW (Hierarchical Navigable Small World) vector search
- 4 distance metrics: Cosine, Euclidean, Manhattan, Dot Product
- LRU frame cache for fast repeated lookups
- Configurable ef_search, ef_construction, max_connections

**Storage**
- SQLite index with WAL mode for concurrent reads
- Incremental append — add documents to existing archives without re-encoding
- Conversation history append for chat memory use cases

## Quick Start

### Prerequisites

- Rust 1.85+
- FFmpeg 8.0+ (`brew install ffmpeg` on macOS)

### Build from Source

```bash
git clone https://github.com/bjenkinsgit/memvid-rs
cd memvid-rs
cargo build --release
```

Feature flags (compile-time):
- `metal` (default) — Metal GPU acceleration on macOS
- `cuda` — CUDA GPU acceleration for NVIDIA

### CLI Usage

```bash
# Encode a document
memvid-rs encode document.pdf -o memory.mp4

# Encode multiple files
memvid-rs encode paper.pdf notes.txt readme.md -o knowledge.mp4

# Search
memvid-rs search "who invented bitcoin" memory.mp4 -k 5

# Append to an existing archive
memvid-rs append memory.mp4 new_document.pdf

# Interactive chat (requires OpenAI API key or local LLM)
memvid-rs chat memory.mp4

# Use a custom config
memvid-rs --config memvid_config.toml encode document.pdf -o memory.mp4
```

### Library Usage

```rust
use memvid_rs::{MemvidEncoder, MemvidRetriever, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Encode
    let mut encoder = MemvidEncoder::new(None).await?;
    encoder.add_pdf("document.pdf").await?;
    encoder.add_text("Additional context", 1024, 32).await?;
    let stats = encoder.build_video("memory.mp4", "index.db").await?;
    println!("Encoded {} chunks", stats.total_chunks);

    // Search
    let mut retriever = MemvidRetriever::new("memory.mp4", "index.db").await?;
    let results = retriever.search("your query", 5).await?;
    for (score, text) in results {
        println!("{:.3}: {}", score, text);
    }

    Ok(())
}
```

### Append to Existing Archives

```rust
let mut encoder = MemvidEncoder::new(None).await?;
encoder.append_document_chunks("memory.mp4", "index.db", "new_doc.pdf").await?;
```

### Pre-computed Embeddings

```rust
// Encode a query to an embedding vector
let embedding = retriever.encode_query("search terms").await?;

// Search by embedding directly (useful for caching or cross-index queries)
let results = retriever.search_by_embedding(&embedding, 5).await?;
```

## Configuration

memvid-rs is configured via TOML files. See `memvid_config.example.toml` for all options.

```toml
[chunking]
chunk_size = 1024        # Characters per chunk
overlap = 32             # Overlap between chunks

[ml]
device = "auto"          # auto | cpu | cuda | metal
model_name = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32

# Remote embedding API (instead of local BERT)
# embedding_api_url = "http://localhost:8000/v1/embeddings"
# embedding_api_model = "text-embedding-3-small"
# embedding_query_prefix = "search_query: "
# embedding_document_prefix = "search_document: "

[qr]
error_correction = "high"   # low | medium | quartile | high
version = 40                # QR version 1-40 (None for auto)
enable_compression = true
compression_threshold = 100

[video]
codec = "prores_ks"         # ProRes — lossless intra-frame for QR
prores_profile = "proxy"    # proxy | lt | standard | hq | 4444 | xq
fps = 30.0
hardware_acceleration = true
library_log_level = "error" # FFmpeg library log level
ffmpeg_cli_log_level = "error"

[search]
engine = "auto"             # auto | hnsw | flat
max_results = 5
min_score_threshold = 0.0

[search.hnsw]
max_connections = 16
ef_construction = 200
ef_search = 50
```

Configuration priority: CLI flags > environment variables > TOML file > defaults.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MEMVID_MODEL_NAME` | Override BERT model (HuggingFace model ID) |
| `EMBEDDING_API_URL` | Remote embedding endpoint (OpenAI-compatible) |
| `FFMPEG_PATH` | Path to FFmpeg binary (if not in PATH) |

## Chat Integration

Works with any OpenAI-compatible API:

```rust
use memvid_rs::{quick_chat, quick_chat_with_config};

// OpenAI
let response = quick_chat("memory.mp4", "index.db", "question", "sk-...").await?;

// Local LLM (Ollama, vLLM, LM Studio, LocalAI)
let response = quick_chat_with_config(
    "memory.mp4", "index.db", "question", "",
    Some("http://localhost:11434/v1"), Some("llama3"),
    None,
).await?;
```

## Architecture

```
Text Documents → Chunking → BERT Embeddings → HNSW Index (SQLite)
                    ↓
              QR Encoding → ProRes Video (.mp4)

Search Query → BERT Embedding → HNSW Lookup → Frame Retrieval → QR Decode → Text
```

**Core modules:**
- `api/` — Public API: `MemvidEncoder`, `MemvidRetriever`, chat functions
- `ml/` — BERT inference via Candle, HNSW indexing, device auto-detection
- `qr/` — QR code encoding (qrcode) and decoding (rqrr) with compression
- `video/` — ProRes video encoding/decoding via FFmpeg, LRU frame cache
- `text/` — Document chunking, PDF extraction
- `storage/` — SQLite index with WAL mode and migrations

## Why ProRes?

Earlier versions used H.265 (HEVC), which caused QR decode failures due to inter-frame compression artifacts. ProRes is an **intra-frame** codec — each frame is independently encoded without referencing other frames. This guarantees lossless QR code preservation on every frame while still providing good compression (ProRes Proxy is ~4:1 for QR content).

On Apple Silicon, ProRes encoding is hardware-accelerated via the dedicated ProRes ASIC, making it faster than software H.265 encoding.

## Files Produced

| File | Contents |
|------|----------|
| `*.mp4` | ProRes video with QR-encoded text chunks |
| `*_index.db` | SQLite database with BERT embeddings and chunk metadata |

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).

## Acknowledgments

- Original [memvid](https://github.com/Olow304/memvid) Python implementation by [Olow304](https://github.com/Olow304)
- [HuggingFace Candle](https://github.com/huggingface/candle) for pure-Rust BERT inference
- [instant-distance](https://github.com/instant-labs/instant-distance) for HNSW vector search
- [qrcode-rs](https://github.com/kennytm/qrcode-rust) and [rqrr](https://github.com/WanzenBug/rqrr) for QR processing
