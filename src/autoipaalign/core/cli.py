"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass, field
import logging
from pathlib import Path

import tyro

from autoipaalign.core.textgrid_io import TextGridContainer, write_textgrids_to_target
from autoipaalign.core.speech_recognition import ASRPipeline


logger = logging.getLogger(__name__)


DEFAULT_TRANSCRIPTION_TIER_NAME = "ipa"
DEFAULT_PHONE_TIER_NAME = "phone"


@dataclass
class OutputConfig:
    """Shared configuration for file output and TextGrid structure."""

    overwrite: bool = False
    """Allow overwriting existing output files."""

    transcription_tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the transcription tier in output TextGrids."""

    enable_phones: bool = False
    """Enable phone alignment tier."""

    phone_tier_name: str = DEFAULT_PHONE_TIER_NAME
    """Name of the phone alignment tier (only used if enable_phones is True)."""


@dataclass
class Transcribe:
    """Transcribe multiple audio files using the desired HuggingFace model.
    New TextGrid files are created and written to the specified
    zip file or output directory.

    Output TextGrids have the same file basename as the corresponding audio files with a .TextGrid suffix.
    """

    audio_paths: list[Path]
    """Paths to audio files to transcribe."""

    output_target: Path
    """Path to directory or zip file to save TextGrid files to."""

    asr: ASRPipeline = field(default_factory=ASRPipeline)
    """Transformers speech recognition pipeline."""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Settings for file output and TextGrid structure."""

    zipped: bool = False
    """Use zipped flag to create a zip file of all TextGrids. Defaults to not zipping."""

    def run(self):
        """Transcribe and write files."""
        if self.output_target.exists():
            if self.output.overwrite:
                logger.warning(
                    "Target %s already exists and may be overwritten.",
                    self.output_target,
                )
            else:
                logger.warning(
                    "Target %s already exists, but cannot be overwritten. Transcriptions may not be saved.",
                    self.output_target,
                )

        logger.info(
            "Transcribing  %s files with model %s.",
            len(self.audio_paths),
            self.asr.model_name,
        )

        text_grids = []

        for audio_path in self.audio_paths:
            tg = TextGridContainer.from_audio_with_predict_transcription(
                audio_path,
                self.output.transcription_tier_name,
                self.asr,
                add_phones=self.output.enable_phones,
                phone_tier_name=self.output.phone_tier_name,
            )
            text_grids.append(tg)

        write_textgrids_to_target(
            self.audio_paths,
            text_grids,
            self.output_target,
            self.zipped,
            self.output.overwrite,
        )


@dataclass
class TranscribeIntervals:
    """Transcribe intervals from an existing TextGrid file using the desired HuggingFace model.
    Interval time frames are taken from the source tier, transcribed, and
    transcriptions are added as intervals in a new target tier.

    Output TextGrids have the same file basename as the corresponding audio files with a .TextGrid suffix and are saved
    in the output_target directory.
    """

    # TODO currently this handles one audio, textgrid pair at a time, but
    # could be made to take multiple paths and pair files

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

    output: OutputConfig = field(default_factory=OutputConfig)
    """Settings for file output and TextGrid structure"""

    def run(self):
        """Execute interval-based transcription."""
        logger.info("Transcribing intervals from %s.", self.textgrid_path)
        self.output_target.mkdir(exist_ok=True, parents=True)

        tg = TextGridContainer.from_textgrid_with_predict_intervals(
            self.audio_path,
            self.textgrid_path,
            self.source_tier,
            self.output.transcription_tier_name,
            self.asr,
            add_phones=self.output.enable_phones,
            phone_tier_name=self.output.phone_tier_name,
        )

        tg.write_textgrid(self.output_target, self.audio_path, self.output.overwrite)


def main():
    """Main entry point for the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(name)s : %(levelname)s : %(message)s")
    cli = tyro.cli(Transcribe | TranscribeIntervals)
    try:
        cli.run()
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    main()
