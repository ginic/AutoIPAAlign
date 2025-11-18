# AutoIPAAlign

Automatically transcribe audio into the International Phonetic Alphabet (IPA) and perform forced alignment. This toolkit includes a command line interface, comparison tools, and interactive web tool.

The AutoIPA project is a collaboration between Virginia Partridge of the UMass Center for Data Science and Artificial
Intelligence and Joe Pater of UMass Linguistics. Its goal is to make automated IPA transcription more useful
to linguists (and others!).
Our first step was to fine-tune a Wav2Vec 2.0 model on the Buckeye corpus, which you can try out here.
Our next steps will be to extend our work to other varieties of English and other languages.
Please reach out to us if you have any questions or comments about our work or have related work to share!
More details are on our [project website](https://websites.umass.edu/comphon/autoipa-automated-ipa-transcription/).

If you use our software, please cite our AMP paper:
Partridge, Virginia, Joe Pater, Parth Bhangla, Ali Nirheche and Brandon Prickett. 2025/to appear. [AI-assisted analysis of phonological variation in English](https://docs.google.com/presentation/d/1IJrfokvX5T_fKkiFXmcYEgRI2ZRwgFU4zU1tNC-iYl0/edit?usp=sharing). Special session on Deep Phonology, AMP 2025, UC Berkeley. To appear in the Proceedings of AMP 2025.
"""

## Basic Usage
This is project is structured in multiple subpackages based on their different external dependencies:
- **autoipaalign.core**: Core library and command-line interface for IPA transcription and forced alignments. Always installed.
- **autoipaalign.compare**: Tools for comparing alignments across different ASR systems, such as whisper and the Montreal Forced Aligner. Install with `pip install autoipaalign[compare]`.
- **autoipaalign.web**: Gradio web interface for interactive transcription. Install with `pip install autoipaalign[compare]`.

### Basic Installation
TODO: Pip install instructions coming soon.

### Command-Line Interface
The `autoipaalign` command lets you transcribe audio and get TextGrid output files with or without forced alignment.
Run `autoipaalign --help` to see the full options.

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
python -m autoipaalign_web.app
```
Then open your browser to the URL shown in the terminal.

## Advanced Usage

### Installing the Development Workspace
This project is structured using [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) based on [this template](https://github.com/konstin/uv-workspace-example-cable/tree/main).

1. Install [uv](https://github.com/astral-sh/uv) if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install to set up development and testing dependencies::
   ```bash
   git clone <repository-url>
   cd autoipaalign
   uv sync --all-extras
   ```

### External Dependencies

-  **Montreal Forced Aligner** (optional, for MFA-based comparisons) should be installed when working with the optional `compare` package.
TODO: update installation instructions for working wiht
   ```bash
   # Install via conda
   conda install -c conda-forge montreal-forced-aligner
   ```

### Comparison Tools

Compare alignments from different ASR systems (documentation coming soon).


### Running Tests

To run unit tests, you can run `uv run pytest` from the root of the repository or inside any of the package subfolders (e.g. `packages/autoipaalign-core`).

### Linting
Linting and formatting checks should pass before any pull requests are merged to the main branch.
Run these checks as follows:

```bash
# From workspace root
uv run ruff check .
uv run ruff format .
```
