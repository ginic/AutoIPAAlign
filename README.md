# AutoIPAAlign

Automatic IPA transcription and forced alignment toolkit with CLI, comparison tools, and web interface.

## Project Structure

This is a UV workspace containing multiple packages:

- **autoipaalign-core**: Core library and command-line interface for IPA transcription
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

## Usage

### Command-Line Interface

```bash
# Transcribe a single audio file
autoipaalign transcribe --audio-paths audio.wav --output-target output/

# Transcribe multiple files to a directory
autoipaalign transcribe --audio-paths audio1.wav audio2.wav --output-target output/

# Transcribe multiple files to a zip file
autoipaalign transcribe --audio-paths audio1.wav audio2.wav --output-target output.zip --zipped

# Transcribe with phone alignment tier
autoipaalign transcribe --audio-paths audio.wav --output-target output/ --output.enable-phones

# Transcribe intervals from existing TextGrid
autoipaalign transcribe-intervals --audio-path audio.wav --textgrid-path existing.TextGrid --source-tier words --output-target output/

# Transcribe intervals with phone alignment tier
autoipaalign transcribe-intervals --audio-path audio.wav --textgrid-path existing.TextGrid --source-tier words --output-target output/ --output.enable-phones

# Use a custom model
autoipaalign transcribe --audio-paths audio.wav --output-target output/ --asr.model-name ginic/full_dataset_train_1_wav2vec2-large-xlsr-53-buckeye-ipa
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

To run unit tests, you can run `uv run pytest` from the root of the repository or inside any of the package subfolders (e.g. `packages/autoipaalign-core`).

### Linting

```bash
# From workspace root
uv run ruff check .
uv run ruff format .
```
