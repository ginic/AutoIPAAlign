"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass, field
import logging
from pathlib import Path

import tyro
import transformers

import autoipaalign_cli.textgrid_io

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"
DEFAULT_TRANSCRIPTION_TIER_NAME = "ipa"
DEFAULT_PHONE_ALIGN_TIER_NAME = "phone"


@dataclass
class TranscriptionConfig:
    model_name: str = field(default=DEFAULT_MODEL, kw_only=True)
    device: int = field(default=-1, kw_only=True)
    model_pipe: transformers.Pipeline = field(init=False)

    def __post_init__(self):
        logger.info("Loading model: %s", self.model_name)
        return transformers.pipeline("automatic-speech-recognition", model=self.model_name, device=self.device)


@dataclass
class TranscribeSingle(TranscriptionConfig):
    """Transcribe a single audio file to IPA."""

    audio_path: Path
    """Path to the audio file to transcribe"""

    output_path: Path | None = None
    """Path to save the TextGrid file (default: audio_path with .TextGrid extension)"""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the tier in the TextGrid"""

    def run(self):
        """Execute single file transcription."""

        logger.info("Transcribing %s with model %s", self.audio_path, self.model_name)
        textgrid = textgrid_io.TextGridContainer.from_audio_and_tra
        print("TODO: Implement single file transcription")


@dataclass
class TranscribeBatch(TranscriptionConfig):
    """Transcribe multiple audio files to IPA."""

    audio_paths: list[Path]
    """Paths to audio files to transcribe"""

    output_dir: Path
    """Directory to save TextGrid files"""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
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
class TranscribeIntervals(TranscriptionConfig):
    """Transcribe intervals from an existing TextGrid."""

    audio_path: Path
    """Path to the audio file"""

    textgrid_path: Path
    """Path to the existing TextGrid file"""

    source_tier: str
    """Name of the source tier containing intervals to transcribe"""

    target_tier: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the new tier to create with IPA transcriptions"""

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
