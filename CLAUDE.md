# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # Run all tests (~1.68s)
cargo test test_name           # Run a single test
cargo clippy                   # Lint
cargo fmt                      # Format code
```

### Feature Flags

- `metal` (default) - Metal GPU acceleration for macOS
- `cuda` - CUDA GPU acceleration for NVIDIA GPUs

```bash
cargo build --release                              # Metal GPU (default on macOS)
cargo build --release --no-default-features        # CPU only
cargo build --release --no-default-features --features cuda  # CUDA
```

GPU is auto-selected at compile time based on features. The `EmbeddingConfig::default()` in `src/ml/embedding.rs` selects:
- Metal when `metal` feature is enabled (default on macOS)
- CUDA when `cuda` feature is enabled
- CPU as fallback

Enable logging to verify device usage:
```bash
RUST_LOG=info cargo run --release ...  # Should show "Using Metal device for TRUE BERT neural network inference"
```

### Examples

```bash
cargo run --example simple_chat   # OpenAI chat example
cargo run --example ollama_chat   # Ollama local LLM example
```

## Architecture

memvid-rs encodes text documents as QR codes in video files, enabling semantic search retrieval.

### Pipeline Flow

```
Text → Chunking → QR Encoding → Video Frames → MP4 File
                     ↓
              BERT Embedding → HNSW Index → SQLite DB

Query → BERT Embedding → HNSW Search → Frame Lookup → QR Decode → Results
```

### Module Structure

- **api/** - Public API layer
  - `MemvidEncoder` - Encodes documents into video + index
  - `MemvidRetriever` - Semantic search and text retrieval
  - `chat` - OpenAI/Ollama chat integration

- **ml/** - Machine learning (pure Rust via Candle)
  - `embedding` - BERT model (sentence-transformers/all-MiniLM-L6-v2)
  - `device` - GPU/CPU auto-detection (singleton DeviceManager)
  - `index` - IndexManager wraps HNSW + chunk metadata
  - `search` - VectorSearchIndex with 4 distance metrics

- **qr/** - QR code processing
  - `encoder` - Text → QR image with compression
  - `decoder` - QR image → text

- **video/** - Video I/O via static FFmpeg
  - `encoder` - Frames → H.265 video
  - `decoder` - Video → frames (with LRU cache)

- **text/** - Document processing
  - `chunking` - Multiple strategies (fixed, sentence, paragraph)
  - `pdf` - PDF text extraction

- **storage/** - SQLite database
  - Schema: chunks, embeddings, frame mappings
  - WAL mode enabled

### Key Patterns

- **Async throughout**: All public APIs are async (Tokio)
- **Result<T> propagation**: Custom `MemvidError` enum with From impls for error chain
- **Test mode**: `#[cfg(test)]` uses fast hash-based embeddings instead of real BERT
- **Configuration**: `Config` struct with Serde support for all modules

### Testing

Tests use hash-based dummy embeddings (defined in ml/embedding.rs under `#[cfg(test)]`) which are ~1000x faster than real BERT inference. Production code uses the actual neural network.

## CLI Usage

```bash
memvid-rs encode document.pdf --output memory.mp4
memvid-rs search "query" --video memory.mp4
memvid-rs chat --video memory.mp4
```
