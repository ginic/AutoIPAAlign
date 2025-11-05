# AutoIPAAlign

Automatic IPA transcription and forced alignment toolkit with CLI, comparison tools, and web interface.

## Project Structure

This is a UV workspace containing multiple packages:

- **autoipaalign-cli**: Core library and command-line interface for IPA transcription
- **autoipaalign-compare**: Tools for comparing alignments across different ASR systems
- **autoipaalign-web**: Gradio web interface for interactive transcription

## Installation

### Prerequisites

#### System Dependencies

1. **ffmpeg** (required for audio processing)
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Montreal Forced Aligner** (optional, for MFA-based comparisons)
   ```bash
   # Install via conda
   conda install -c conda-forge montreal-forced-aligner
   ```

### Installing the Workspace

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install:
   ```bash
   git clone <repository-url>
   cd autoipaalign
   uv sync --all-packages
   ```

   To set up development and testing dependencies:
   2. Clone the repository and install:
   ```bash
   git clone <repository-url>
   cd autoipaalign
   uv sync --all-packages --all-extras
   ```

### Installing Individual Packages

You can install specific packages as needed:

```bash
# CLI tool only
uv pip install -e packages/autoipaalign-cli

# With comparison tools
uv pip install -e packages/autoipaalign-compare

# With web interface
uv pip install -e packages/autoipaalign-web

# With optional Whisper support for comparisons
uv pip install -e "packages/autoipaalign-compare[whisper]"
```

## Usage

### Command-Line Interface

```bash
# Transcribe a single audio file
autoipaalign transcribe-single audio.wav --model-name ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa

# Transcribe multiple files
autoipaalign transcribe-batch audio1.wav audio2.wav --output-dir output/

# Transcribe intervals from existing TextGrid
autoipaalign transcribe-intervals audio.wav existing.TextGrid --source-tier words --target-tier IPA
```

### Web Interface

```bash
cd packages/autoipaalign-web
uv run python -m autoipaalign_web.app
```

Then open your browser to the URL shown in the terminal.

### Comparison Tools

Compare alignments from different ASR systems (documentation coming soon).

## Development

### Running Tests

```bash
# Test CLI package
cd packages/autoipaalign-cli
uv run pytest

# Test comparison package
cd packages/autoipaalign-compare
uv run pytest
```

### Linting

```bash
# From workspace root
uv run ruff check .
uv run ruff format .
```

## Available Models

The default model is `ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa`.

See the full list of available models in the [models documentation](packages/autoipaalign-cli/src/autoipaalign_cli/cli.py).
