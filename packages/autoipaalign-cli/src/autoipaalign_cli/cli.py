"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass
from pathlib import Path

import tyro


DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"


@dataclass
class TranscribeSingle:
    """Transcribe a single audio file to IPA."""

    audio_path: Path
    """Path to the audio file to transcribe"""

    output_path: Path | None = None
    """Path to save the TextGrid file (default: audio_path with .TextGrid extension)"""

    model_name: str = DEFAULT_MODEL
    """HuggingFace model identifier for ASR"""

    tier_name: str = "IPA"
    """Name of the tier in the TextGrid"""

    def run(self):
        """Execute single file transcription."""
        print(f"Transcribing {self.audio_path} with model {self.model_name}")
        print(f"Output tier: {self.tier_name}")
        # TODO: Implement transcription logic
        print("TODO: Implement single file transcription")


@dataclass
class TranscribeBatch:
    """Transcribe multiple audio files to IPA."""

    audio_paths: list[Path]
    """Paths to audio files to transcribe"""

    output_dir: Path
    """Directory to save TextGrid files"""

    model_name: str = DEFAULT_MODEL
    """HuggingFace model identifier for ASR"""

    tier_name: str = "IPA"
    """Name of the tier in the TextGrid"""

    create_zip: bool = False
    """Create a zip file of all TextGrids"""

    def run(self):
        """Execute batch transcription."""
        print(f"Transcribing {len(self.audio_paths)} files with model {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Output tier: {self.tier_name}")
        print(f"Create zip: {self.create_zip}")
        # TODO: Implement batch transcription logic
        print("TODO: Implement batch transcription")


@dataclass
class TranscribeIntervals:
    """Transcribe intervals from an existing TextGrid."""

    audio_path: Path
    """Path to the audio file"""

    textgrid_path: Path
    """Path to the existing TextGrid file"""

    source_tier: str
    """Name of the source tier containing intervals to transcribe"""

    target_tier: str = "IPATier"
    """Name of the new tier to create with IPA transcriptions"""

    model_name: str = DEFAULT_MODEL
    """HuggingFace model identifier for ASR"""

    output_path: Path | None = None
    """Path to save the output TextGrid (default: audio_path with _IPA.TextGrid suffix)"""

    def run(self):
        """Execute interval-based transcription."""
        print(f"Transcribing intervals from {self.textgrid_path}")
        print(f"Audio: {self.audio_path}")
        print(f"Source tier: {self.source_tier} -> Target tier: {self.target_tier}")
        print(f"Model: {self.model_name}")
        # TODO: Implement interval transcription logic
        print("TODO: Implement interval transcription")


@dataclass
class CLI:
    """Automatic IPA transcription and forced alignment CLI."""

    command: TranscribeSingle | TranscribeBatch | TranscribeIntervals
    """The command to execute"""

    def run(self):
        """Execute the selected command."""
        self.command.run()


def main():
    """Main entry point for the CLI."""
    cli = tyro.cli(CLI)
    cli.run()


if __name__ == "__main__":
    main()
