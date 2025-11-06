"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import tyro
import transformers

from autoipaalign_cli.textgrid_io import TextGridContainer, write_textgrids_to_target

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"
DEFAULT_TRANSCRIPTION_TIER_NAME = "ipa"
DEFAULT_PHONE_ALIGN_TIER_NAME = "phone"


@dataclass
class ASRPipeline:
    """Handles loading and configuration of the Transformer pipeline"""

    model_name: str = field(default=DEFAULT_MODEL)
    """The name of the HuggingFace model used to transcribe speech. """

    device: int | str = field(default=-1)
    """Index of the device for model inference. Defaults to -1 for CPU.
    """

    _model_pipe: transformers.Pipeline = field(init=False)

    def __post_init__(self):
        logger.info("Loading model: %s", self.model_name)
        self._model_pipe = transformers.pipeline(
            "automatic-speech-recognition", model=self.model_name, device=self.device
        )


@dataclass
class TranscriptionConfig:
    asr_pipeline: ASRPipeline
    """Transformers speech recognition pipeline
    """

    sampling_rate: int = field(default=16000, kw_only=True)
    """Sampling rate for audio preprocessing. Defaults to 16K."""

    overwrite: bool = field(default=False, kw_only=True)
    """Use this flag to allow overwriting existing TextGrid files."""


@dataclass
class Transcribe(TranscriptionConfig):
    """Transcribe multiple audio files using the desired HuggingFace model.
    New TextGrid files are created and written to the specified
    zip file or output directory.

    Output TextGrids have the filename as the corresponding audio files with a .TextGrid suffix.
    """

    audio_paths: list[Path]
    """Paths to audio files to transcribe"""

    output_target: Path
    """Name of directory or zip file to save TextGrid files to."""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the tier in the TextGrid"""

    zipped: bool = False
    """Use flag to create a zip file of all TextGrids"""

    def run(self):
        """Transcribe and write files."""
        if self.output_target.exists():
            if self.overwrite:
                logger.warning("Target %s already exists and may be overwritten.")
            else:
                logger.warning("Target %s already exists, but cannot be overwritten. Transcriptions may not be saved.")

        logger.info("Transcribing  %s files with model %s.", len(self.audio_paths), self.asr_pipeline.model_name)

        transcriptions = self.asr_pipeline._model_pipe(self.audio_paths)
        text_grids = []

        for audio_path, transcription in zip(self.audio_paths, transcriptions):
            tg = TextGridContainer.from_audio_and_transcription(audio_path, self.tier_name, transcription["text"])
            text_grids.append(tg)

        write_textgrids_to_target(self.audio_paths, text_grids, self.output_target, self.zipped, self.overwrite)


@dataclass
class TranscribeIntervals(TranscriptionConfig):
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

    target_tier: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the new tier to create with IPA transcriptions"""

    def run(self):
        """Execute interval-based transcription."""
        logger.info("Transcribing intervals from %s.", self.textgrid_path)
        tg = TextGridContainer.from_textgrid_with_predict_intervals(
            self.audio_path,
            self.textgrid_path,
            self.source_tier,
            self.target_tier,
            self.asr_pipeline._model_pipe,
            self.sampling_rate,
        )
        tg.write_textgrid(self.output_target, self.audio_path, self.overwrite)


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
