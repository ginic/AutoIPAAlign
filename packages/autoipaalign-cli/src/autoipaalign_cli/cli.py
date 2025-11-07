"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import tyro

from autoipaalign_cli.textgrid_io import TextGridContainer, write_textgrids_to_target
from autoipaalign_cli.speech_recognition import ASRPipeline


logger = logging.getLogger(__name__)


DEFAULT_TRANSCRIPTION_TIER_NAME = "ipa"
DEFAULT_PHONE_ALIGN_TIER_NAME = "phone"


@dataclass
class IOConfig:
    """Shared behaviour for audio and file input/output used
    during the transcription process.
    """

    # TODO Configure codec options here in the future

    overwrite: bool = field(default=False, kw_only=True)
    """Use this flag to allow overwriting existing output files."""


@dataclass
class Transcribe:
    """Transcribe multiple audio files using the desired HuggingFace model.
    New TextGrid files are created and written to the specified
    zip file or output directory.

    Output TextGrids have the filename as the corresponding audio files with a .TextGrid suffix.
    """

    audio_paths: list[Path]
    """Paths to audio files to transcribe"""

    output_target: Path
    """Name of directory or zip file to save TextGrid files to."""

    asr: ASRPipeline = field(default_factory=ASRPipeline)
    """Transformers speech recognition pipeline"""

    io: IOConfig = field(default_factory=IOConfig)
    """Settings audio and file input/output"""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the tier in the TextGrid"""

    zipped: bool = False
    """Use flag to create a zip file of all TextGrids"""

    def run(self):
        """Transcribe and write files."""
        if self.output_target.exists():
            if self.io.overwrite:
                logger.warning("Target %s already exists and may be overwritten.")
            else:
                logger.warning("Target %s already exists, but cannot be overwritten. Transcriptions may not be saved.")

        logger.info("Transcribing  %s files with model %s.", len(self.audio_paths), self.asr.model_name)

        text_grids = []

        for audio_path in self.audio_paths:
            tg = TextGridContainer.from_audio_with_predict_transcription(audio_path, self.tier_name, self.asr)
            text_grids.append(tg)

        write_textgrids_to_target(self.audio_paths, text_grids, self.output_target, self.zipped, self.io.overwrite)


@dataclass
class TranscribeIntervals:
    """Transcribe intervals from an existing TextGrid file using the desired HuggingFace model.
    Interval time frames are taken from the source tier, transcribed, and
    transcriptions are added as intervals in a new target tier.

    Output TextGrids have the filename as the corresponding audio files with a .TextGrid suffix.
    """

    audio_path: Path
    """Path to the audio file"""

    textgrid_path: Path
    """Path to the existing TextGrid file"""

    output_target: Path
    """Name of directory to save TextGrid files to."""

    source_tier: str
    """Name of the source tier containing intervals to transcribe"""

    asr: ASRPipeline = field(default_factory=ASRPipeline)
    """Transformers speech recognition pipeline"""

    io: IOConfig = field(default_factory=IOConfig)
    """Settings audio and file input/output"""

    target_tier: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the new tier to create with IPA transcriptions"""

    def run(self):
        """Execute interval-based transcription."""
        logger.info("Transcribing intervals from %s.", self.textgrid_path)
        tg = TextGridContainer.from_textgrid_with_predict_intervals(
            self.audio_path, self.textgrid_path, self.source_tier, self.target_tier, self.asr
        )
        tg.write_textgrid(self.output_target, self.audio_path, self.io.overwrite)


def main():
    """Main entry point for the CLI."""
    cli = tyro.cli(Transcribe | TranscribeIntervals)
    try:
        cli.run()
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    main()
