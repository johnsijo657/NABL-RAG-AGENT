import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Define local directory inside the project
project_root = Path(__file__).parent.parent.absolute()
models_dir = project_root / "local_models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"

# Create directories if they don't exist
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading model directly to: {models_dir}")
print("This guarantees offline reliability and prevents version conflicts.")

# Download the model files directly into the project folder (no symlinks)
snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir=str(models_dir),
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot"] # Skip unnecessary formats to save space
)

print("Download successfully complete. Your model is now 100% self-contained.")
