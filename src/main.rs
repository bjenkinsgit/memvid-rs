//! memvid-rs CLI application
//!
//! Command-line interface for the memvid-rs library.

use clap::{Parser, Subcommand};
use memvid_rs::{Config, MemvidEncoder, MemvidRetriever};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "memvid-rs")]
#[command(
    about = "A high-performance QR code video encoder for text storage and semantic retrieval"
)]
#[command(version)]
struct Cli {
    /// Path to TOML config file (e.g. memvid_config.toml)
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Remote embedding API URL (overrides config file and EMBEDDING_API_URL env var)
    #[arg(long, global = true)]
    embedding_api_url: Option<String>,

    /// Remote embedding model name (overrides config file and EMBEDDING_API_MODEL env var)
    #[arg(long, global = true)]
    embedding_api_model: Option<String>,

    /// Query prefix for asymmetric/instruction-tuned embedding models
    #[arg(long, global = true)]
    query_prefix: Option<String>,

    /// Document prefix for asymmetric/instruction-tuned embedding models
    #[arg(long, global = true)]
    document_prefix: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode documents into a QR code video
    Encode {
        /// Input file(s) to encode
        #[arg(required = true)]
        inputs: Vec<PathBuf>,

        /// Output video file
        #[arg(short, long, default_value = "memory.mp4")]
        output: PathBuf,

        /// Output index file (SQLite database)
        #[arg(short, long, default_value = "memory_index.db")]
        index: PathBuf,

        /// Chunk size in characters
        #[arg(long, default_value = "1024")]
        chunk_size: usize,

        /// Overlap between chunks
        #[arg(long, default_value = "32")]
        overlap: usize,
    },

    /// Search within a QR code video
    Search {
        /// Video file to search
        #[arg(short, long)]
        video: PathBuf,

        /// Index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,

        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
    },

    /// Interactive chat with your documents
    Chat {
        /// Video file
        #[arg(short, long)]
        video: PathBuf,

        /// Index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,
    },

    /// Add new content to existing knowledge base (incremental update)
    Append {
        /// Existing video file to append to
        #[arg(short, long)]
        video: PathBuf,

        /// Existing index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,

        /// New input file(s) to add
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },

    /// Store LLM conversation history to knowledge base
    AppendConversation {
        /// Existing video file to append to
        #[arg(short, long)]
        video: PathBuf,

        /// Existing index file (SQLite database)
        #[arg(short, long)]
        index: PathBuf,

        /// Conversation history file (JSON format: [{"human": "...", "assistant": "..."}])
        #[arg(short, long)]
        conversation_file: PathBuf,
    },
}

/// Build a Config with priority: CLI flags > env vars > config file > defaults.
fn build_config(cli: &Cli) -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = match &cli.config {
        Some(path) => {
            eprintln!("Loading config from: {}", path.display());
            Config::from_toml_file(path)?
        }
        None => Config::default(),
    };

    // Env var fallbacks (only if not already set by config file)
    if config.ml.embedding_api_url.is_none() {
        if let Ok(url) = std::env::var("EMBEDDING_API_URL") {
            config.ml.embedding_api_url = Some(url);
        }
    }
    if config.ml.embedding_api_model.is_none() {
        if let Ok(model) = std::env::var("EMBEDDING_API_MODEL") {
            config.ml.embedding_api_model = Some(model);
        }
    }

    // CLI flags override everything
    if let Some(ref url) = cli.embedding_api_url {
        config.ml.embedding_api_url = Some(url.clone());
    }
    if let Some(ref model) = cli.embedding_api_model {
        config.ml.embedding_api_model = Some(model.clone());
    }
    if let Some(ref prefix) = cli.query_prefix {
        config.ml.embedding_query_prefix = Some(prefix.clone());
    }
    if let Some(ref prefix) = cli.document_prefix {
        config.ml.embedding_document_prefix = Some(prefix.clone());
    }

    Ok(config)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    let cli = Cli::parse();
    let config = build_config(&cli)?;

    match cli.command {
        Commands::Encode {
            inputs,
            output,
            index,
            chunk_size,
            overlap,
        } => {
            encode_command(inputs, output, index, chunk_size, overlap, config).await?;
        }
        Commands::Search {
            video,
            index,
            query,
            top_k,
        } => {
            search_command(video, index, query, top_k, config).await?;
        }
        Commands::Chat { video, index } => {
            chat_command(video, index, config).await?;
        }
        Commands::Append {
            video,
            index,
            inputs,
        } => {
            append_command(video, index, inputs, config).await?;
        }
        Commands::AppendConversation {
            video,
            index,
            conversation_file,
        } => {
            append_conversation_command(video, index, conversation_file, config).await?;
        }
    }

    Ok(())
}

async fn encode_command(
    inputs: Vec<PathBuf>,
    output: PathBuf,
    index: PathBuf,
    chunk_size: usize,
    overlap: usize,
    mut config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting memvid encoding...");

    config.chunking.chunk_size = chunk_size;
    config.chunking.overlap = overlap;

    let mut encoder = MemvidEncoder::new(Some(config)).await?;

    for input in inputs {
        println!("📄 Processing: {}", input.display());

        if !input.exists() {
            eprintln!("❌ File not found: {}", input.display());
            continue;
        }

        match input.extension().and_then(|ext| ext.to_str()) {
            Some("pdf") => {
                encoder.add_pdf(&input).await?;
            }
            Some("txt") | Some("md") | Some("markdown") => {
                encoder.add_text_file(&input).await?;
            }
            _ => {
                // Try to read as text file
                match encoder.add_text_file(&input).await {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("❌ Failed to process {}: {}", input.display(), e);
                        continue;
                    }
                }
            }
        }
    }

    if encoder.chunk_count() == 0 {
        eprintln!("❌ No content was successfully processed");
        return Ok(());
    }

    println!("🔧 Building video with {} chunks...", encoder.chunk_count());

    let stats = encoder
        .build_video(output.to_str().unwrap(), index.to_str().unwrap())
        .await?;

    println!("✅ Encoding complete!");
    println!("   📊 Chunks: {}", stats.total_chunks);
    println!("   🎞️  Frames: {}", stats.total_frames);
    println!("   ⏱️  Time: {:.2}s", stats.processing_time);
    println!("   📹 Video: {}", output.display());
    println!("   📋 Index: {}", index.display());

    Ok(())
}

async fn search_command(
    video: PathBuf,
    index: PathBuf,
    query: String,
    top_k: usize,
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Searching for: \"{}\"", query);

    let mut retriever = MemvidRetriever::new_with_config(&video, &index, Some(config)).await?;
    let results = retriever.search(&query, top_k).await?;

    if results.is_empty() {
        println!("❌ No results found");
        return Ok(());
    }

    println!("📋 Found {} results:", results.len());
    println!();

    for (i, (score, text)) in results.iter().enumerate() {
        println!("{}. Score: {:.3}", i + 1, score);
        println!("   {}", text);
        println!();
    }

    Ok(())
}

async fn chat_command(
    video: PathBuf,
    index: PathBuf,
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("💬 Starting interactive chat mode...");
    println!("   Type 'quit' or 'exit' to end the session");
    println!();

    let mut retriever = MemvidRetriever::new_with_config(&video, &index, Some(config)).await?;

    loop {
        print!("❓ Query: ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("👋 Goodbye!");
            break;
        }

        let results = retriever.search(input, 3).await?;

        if results.is_empty() {
            println!("❌ No results found for: \"{}\"", input);
        } else {
            println!("📋 Results:");
            for (i, (score, text)) in results.iter().enumerate() {
                println!("{}. (Score: {:.3}) {}", i + 1, score, text);
            }
        }
        println!();
    }

    Ok(())
}

async fn append_command(
    video: PathBuf,
    index: PathBuf,
    inputs: Vec<PathBuf>,
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting incremental update...");

    // Verify existing files exist
    if !video.exists() {
        eprintln!("❌ Existing video file not found: {}", video.display());
        return Ok(());
    }
    if !index.exists() {
        eprintln!("❌ Existing index file not found: {}", index.display());
        return Ok(());
    }

    let mut encoder = MemvidEncoder::new(Some(config)).await?;
    let mut total_added_chunks = 0;
    let mut total_processing_time = 0.0;

    for input in inputs {
        println!("📄 Processing: {}", input.display());

        if !input.exists() {
            eprintln!("❌ File not found: {}", input.display());
            continue;
        }

        let start_time = std::time::Instant::now();

        // Use append_document_chunks for each file - this handles the incremental update correctly
        let stats = match encoder
            .append_document_chunks(
                video.to_str().unwrap(),
                index.to_str().unwrap(),
                input.to_str().unwrap(),
            )
            .await
        {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("❌ Failed to process {}: {}", input.display(), e);
                continue;
            }
        };

        total_added_chunks += stats.total_chunks;
        total_processing_time += stats.processing_time;

        println!(
            "   ✅ Added {} chunks from {} in {:.2}s",
            stats.total_chunks,
            input.display(),
            start_time.elapsed().as_secs_f64()
        );
    }

    if total_added_chunks == 0 {
        eprintln!("❌ No content was successfully processed");
        return Ok(());
    }

    println!("✅ Incremental update complete!");
    println!("   📊 Total added chunks: {}", total_added_chunks);
    println!("   🎞️  Total added frames: {}", total_added_chunks);
    println!("   ⏱️  Total time: {:.2}s", total_processing_time);
    println!("   📹 Updated video: {}", video.display());
    println!("   📋 Updated index: {}", index.display());

    Ok(())
}

async fn append_conversation_command(
    video: PathBuf,
    index: PathBuf,
    conversation_file: PathBuf,
    config: Config,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎬 Starting conversation history append...");

    // Verify existing files exist
    if !video.exists() {
        eprintln!("❌ Existing video file not found: {}", video.display());
        return Ok(());
    }
    if !index.exists() {
        eprintln!("❌ Existing index file not found: {}", index.display());
        return Ok(());
    }

    if !conversation_file.exists() {
        eprintln!(
            "❌ Conversation history file not found: {}",
            conversation_file.display()
        );
        return Ok(());
    }

    let mut encoder = MemvidEncoder::new(Some(config)).await?;

    println!("📄 Processing conversation history file...");

    // Try to parse as JSON conversation format first
    let file_content = std::fs::read_to_string(&conversation_file)?;

    // TODO: For now, parse JSON conversations
    // Expected format: [{"human": "...", "assistant": "..."}, ...]
    if let Ok(json_conversations) = serde_json::from_str::<Vec<serde_json::Value>>(&file_content) {
        let mut conversations = Vec::new();

        for conv in json_conversations {
            if let (Some(human), Some(assistant)) = (
                conv.get("human").and_then(|v| v.as_str()),
                conv.get("assistant").and_then(|v| v.as_str()),
            ) {
                conversations.push((human.to_string(), assistant.to_string()));
            }
        }

        if !conversations.is_empty() {
            let stats = encoder
                .append_conversation_history(
                    video.to_str().unwrap(),
                    index.to_str().unwrap(),
                    conversations,
                )
                .await?;

            println!("✅ Conversation history append complete!");
            println!("   💬 Conversation turns: {}", stats.total_chunks / 2);
            println!("   📊 Total chunks: {}", stats.total_chunks);
            println!("   🎞️  Total frames: {}", stats.total_frames);
            println!("   ⏱️  Time: {:.2}s", stats.processing_time);
            println!("   📹 Updated video: {}", video.display());
            println!("   📋 Updated index: {}", index.display());
        } else {
            eprintln!("❌ No valid conversations found in JSON file");
        }
    } else {
        // Fallback: treat as plain text file
        println!("📄 JSON parsing failed, treating as plain text file...");
        let stats = encoder
            .append_document_chunks(
                video.to_str().unwrap(),
                index.to_str().unwrap(),
                conversation_file.to_str().unwrap(),
            )
            .await?;

        println!("✅ Conversation history append complete!");
        println!("   📊 Chunks: {}", stats.total_chunks);
        println!("   🎞️  Frames: {}", stats.total_frames);
        println!("   ⏱️  Time: {:.2}s", stats.processing_time);
        println!("   📹 Updated video: {}", video.display());
        println!("   📋 Updated index: {}", index.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test that CLI parsing works
        let cli = Cli::try_parse_from(&["memvid-rs", "encode", "test.txt"]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_global_config_flag() {
        let cli = Cli::try_parse_from(&[
            "memvid-rs",
            "--config",
            "test.toml",
            "search",
            "-v",
            "video.mp4",
            "-i",
            "index.db",
            "test query",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(cli.config, Some(PathBuf::from("test.toml")));
    }

    #[test]
    fn test_cli_embedding_flags() {
        let cli = Cli::try_parse_from(&[
            "memvid-rs",
            "--embedding-api-url",
            "http://localhost:8000/v1/embeddings",
            "--embedding-api-model",
            "e5-mistral",
            "--query-prefix",
            "Instruct: Retrieve relevant passages\nQuery: ",
            "--document-prefix",
            "passage: ",
            "search",
            "-v",
            "video.mp4",
            "-i",
            "index.db",
            "test query",
        ]);
        assert!(cli.is_ok());
        let cli = cli.unwrap();
        assert_eq!(
            cli.embedding_api_url,
            Some("http://localhost:8000/v1/embeddings".to_string())
        );
        assert_eq!(
            cli.embedding_api_model,
            Some("e5-mistral".to_string())
        );
        assert_eq!(
            cli.query_prefix,
            Some("Instruct: Retrieve relevant passages\nQuery: ".to_string())
        );
        assert_eq!(cli.document_prefix, Some("passage: ".to_string()));
    }

    #[test]
    fn test_build_config_defaults() {
        let cli = Cli::try_parse_from(&["memvid-rs", "encode", "test.txt"]).unwrap();
        let config = build_config(&cli).unwrap();
        assert_eq!(
            config.ml.model_name,
            std::env::var("MEMVID_MODEL_NAME")
                .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string())
        );
        assert!(config.ml.embedding_api_url.is_none() || std::env::var("EMBEDDING_API_URL").is_ok());
    }

    #[test]
    fn test_build_config_cli_overrides() {
        let cli = Cli::try_parse_from(&[
            "memvid-rs",
            "--embedding-api-url",
            "http://test:8000/v1/embeddings",
            "--query-prefix",
            "query: ",
            "encode",
            "test.txt",
        ])
        .unwrap();
        let config = build_config(&cli).unwrap();
        assert_eq!(
            config.ml.embedding_api_url,
            Some("http://test:8000/v1/embeddings".to_string())
        );
        assert_eq!(
            config.ml.embedding_query_prefix,
            Some("query: ".to_string())
        );
    }
}
