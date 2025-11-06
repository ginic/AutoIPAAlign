"""Command-line interface for automatic IPA transcription and forced alignment."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import zipfile

import tyro
import transformers

from autoipaalign_cli.textgrid_io import TextGridContainer, to_textgrid_basename

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"
DEFAULT_TRANSCRIPTION_TIER_NAME = "ipa"
DEFAULT_PHONE_ALIGN_TIER_NAME = "phone"


@dataclass
class ASRPipeline:
    """Handles loading and configuration of the Transformer pipeline"""

    model_name: str = field(default=DEFAULT_MODEL)
    """The name of the HuggingFace model used to transcribe speech. """

    device: int = field(default=-1)
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


@dataclass
class TranscribeSingle(TranscriptionConfig):
    """Transcribe a single audio file to IPA."""

    audio_path: Path
    """Path to the audio file to transcribe"""

    output_dir: Path
    """Directory to save the TextGrid file named like the audio file, but with a .TextGrid extension)"""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the tier in the TextGrid"""

    def transcribe(self) -> TextGridContainer:
        logger.info("Transcribing %s with model %s", self.audio_path, self.asr_pipeline.model_name)

        tg = TextGridContainer.from_audio_with_predict_transcription(
            self.audio_path, self.tier_name, self.asr_pipeline._model_pipe, self.sampling_rate
        )
        return tg

    def run(self):
        """Transcribe a single audio and save the result as a TextGrid file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tg = self.transcribe()
        return tg.write_textgrid(self.output_dir, self.audio_path)


@dataclass
class TranscribeBatch(TranscriptionConfig):
    """Transcribe multiple audio files to IPA."""

    audio_paths: list[Path]
    """Paths to audio files to transcribe"""

    output_target: Path
    """Name of directory or zip file to save TextGrid files to"""

    tier_name: str = DEFAULT_TRANSCRIPTION_TIER_NAME
    """Name of the tier in the TextGrid"""

    zipped: bool = False
    """Use flag to create a zip file of all TextGrids"""

    def run(self):
        """Transcribe and write files."""
        logger.info(f"Transcribing {len(self.audio_paths)} files with model {self.asr_pipeline.model_name}")
        transcriptions = self.asr_pipeline._model_pipe(self.audio_paths)
        text_grids = []

        for audio_path, transcription in zip(self.audio_paths, transcriptions):
            tg = TextGridContainer.from_audio_and_transcription(audio_path, self.tier_name, transcription["text"])
            text_grids.append(tg)

        if self.zipped:
            logger.info("Writing TextGrids to zip file %s", self.output_target)
            with zipfile.ZipFile(self.output_target, "w") as zipf:
                for i, (audio_path, tg) in enumerate(zip(self.audio_paths, text_grids), start=1):
                    zipf.writestr(to_textgrid_basename(audio_path), tg.export_to_long_textgrid_str())
                    if i % 10 == 0:
                        logger.info("%s TextGrids written to zip", i)

        else:
            if not self.output_target.exists():
                logger.info("Making output directory %s", self.output_target)
                self.output_target.mkdir(parents=True)
            logger.info("Writing TextGrids to %s", self.output_target)
            for i, (audio_path, tg) in enumerate(zip(self.audio_paths, text_grids), start=1):
                tg.write_textgrid(self.output_target, audio_path)
                if i % 10 == 0:
                    logger.info("%s TextGrids written", i)


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
