"""Utilities for manipulating TextGrid files with audio and transcriptions.

This module provides a container class for working with Praat TextGrid files,
including reading from and writing to files, creating TextGrids from audio and
transcriptions, and generating new tiers using automatic speech recognition (ASR).
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import warnings
import zipfile

import librosa
import tgt.core
import tgt.io3
import transformers

logger = logging.getLogger(__name__)

TEXT_GRID_SUFFIX = ".TextGrid"


def to_textgrid_basename(filename: Path):
    return filename.with_suffix(TEXT_GRID_SUFFIX).name


def write_textgrids_to_target(
    audio_paths: list[Path],
    text_grids: list["TextGridContainer"],
    target_path: Path,
    is_zip: bool = False,
    is_overwrite: bool = True,
):
    """Write multiple TextGrids to a directory or zip file. Existing zip file or TextGrid files will be overwritten.

    Args:
        audio_paths: List of paths to audio files corresponding to the TextGrids.
            Used to generate TextGrid filenames.
        text_grids: List of TextGridContainer objects to write contents of.
        target_path: Destination path - either a directory or a zip file path.
        is_zip: If True, write TextGrids to a zip file at target_path.
            If False, write individual TextGrid files to the target_path directory.
            Defaults to False.
        is_overwrite: Boolean flag, allow overwriting existing files or not.
    """
    if is_zip:
        logger.info("Writing TextGrids to zip file %s", target_path)
        if target_path.exists() and not is_overwrite:
            raise OSError(f"{target_path} already exists and cannot be overwritten")

        with zipfile.ZipFile(target_path, "w") as zipf:
            for i, (audio_path, tg) in enumerate(zip(audio_paths, text_grids), start=1):
                zipf.writestr(to_textgrid_basename(audio_path), tg.export_to_long_textgrid_str())
                if i % 10 == 0:
                    logger.info("%s TextGrids written to zip", i)
    else:
        if not target_path.exists():
            logger.info("Making output directory %s", target_path)
            target_path.mkdir(parents=True)
        logger.info("Writing TextGrids to %s", target_path)
        for i, (audio_path, tg) in enumerate(zip(audio_paths, text_grids), start=1):
            tg.write_textgrid(target_path, audio_path, is_overwrite)
            if i % 10 == 0:
                logger.info("%s TextGrids written", i)


@dataclass
class TextGridContainer:
    """Container for TextGrid objects with utilities for I/O and manipulation.

    This class wraps a tgt.core.TextGrid object and provides methods for
    exporting, reading, writing, and creating TextGrids from various sources
    including audio files and ASR predictions.

    Attributes:
        text_grid: The underlying tgt.core.TextGrid object.
    """

    text_grid: tgt.core.TextGrid

    def export_to_long_textgrid_str(self) -> str:
        """Export the TextGrid to a long-format string representation.

        Returns:
            A string containing the TextGrid in Praat long text format.
        """
        return tgt.io3.export_to_long_textgrid(self.text_grid)

    def get_tier_names(self) -> list[str]:
        """Get the names of all tiers in the TextGrid.

        Returns:
            A list of tier names as strings.
        """
        return self.text_grid.get_tier_names()

    def write_textgrid(self, directory: Path, filename: Path, is_overwrite: bool = True) -> Path:
        """Write the TextGrid to a file in the specified directory using hte same

        Args:
            directory: The directory where the TextGrid file will be written.
            filename: Path or filename with to use as the basename for the output.
                Usually the path to the corresponding audio file.
            is_overwrite: Boolean flag, allow overwriting existing files or not.

        Returns:
            The full path to the written TextGrid file.
        """
        textgrid_path = Path(directory) / to_textgrid_basename(filename)

        if is_overwrite and textgrid_path.exists():
            raise OSError("File %s already exists and cannot be overwritten", textgrid_path)

        logger.debug("Writing TextGrid to %s", textgrid_path)
        textgrid_path.write_text(self.export_to_long_textgrid_str())
        logger.debug("TextGrid %s written", textgrid_path)
        return textgrid_path

    def validate_against_audio_duration(self, audio_path: str | os.PathLike[str], time_difference=0.01):
        audio_duration = librosa.get_duration(path=audio_path, sr=None)
        tg_end_time = max(tier.end_time for tier in self.text_grid.tiers)
        if tg_end_time > audio_duration:
            raise ValueError("TextGrid ends at {tg_end_time:.2f}s but audio is only {audio_duration:.2f}s.")

        if abs(tg_end_time - audio_duration) > time_difference:
            warning = f"TextGrid ends at {tg_end_time:.2f}s but audio is {audio_duration:.2f}s. Only the annotated portion will be transcribed."
            warnings.warn(warning)  # So this appears in gradio
            logger.warning(warning)

    @classmethod
    def from_textgrid_file(cls, textgrid_file: Path) -> "TextGridContainer":
        """Create a TextGridContainer from an existing TextGrid file.

        Args:
            textgrid_file: Path to the TextGrid file to read.

        Returns:
            A new TextGridContainer instance containing the loaded TextGrid.
        """
        tg = tgt.io3.read_textgrid(textgrid_file)
        return cls(text_grid=tg)

    @classmethod
    def from_audio_with_predict_transcription(
        cls,
        audio_in: str | os.PathLike[str],
        textgrid_tier_name: str,
        asr_pipeline: transformers.Pipeline,
        sampling_rate: int = 16000,
    ) -> "TextGridContainer":
        try:
            y, sr = librosa.load(audio_in, sr=sampling_rate)
            transcription = asr_pipeline(y)["text"]
        except Exception as e:
            transcription = f"[Error]: {e}"
        return cls.from_audio_and_transcription(audio_in, textgrid_tier_name, transcription)

    @classmethod
    def from_audio_and_transcription(
        cls, audio_in: str | os.PathLike[str], textgrid_tier_name: str, transcription: str
    ) -> "TextGridContainer":
        """Create a TextGrid with a single tier from audio and transcription.

        The transcription is added as a single interval spanning the entire
        audio duration.

        Args:
            audio_in: Path to the audio file.
            textgrid_tier_name: Desired name for the transcription's tier.
            transcription: Transcription text to add as the interval value.

        Returns:
            A new TextGridContainer with a single tier containing the transcription.
            Returns an empty TextGridContainer if audio_in or transcription is None.
        """
        if audio_in is None or transcription is None:
            return cls(text_grid=tgt.core.TextGrid())

        duration = librosa.get_duration(path=audio_in, sr=None)

        annotation = tgt.core.Interval(0, duration, transcription)
        transcription_tier = tgt.core.IntervalTier(start_time=0, end_time=duration, name=textgrid_tier_name)
        transcription_tier.add_annotation(annotation)
        textgrid = tgt.core.TextGrid()
        textgrid.add_tier(transcription_tier)
        return cls(text_grid=textgrid)

    @classmethod
    def from_textgrid_with_predict_intervals(
        cls,
        audio_in: str | os.PathLike[str],
        textgrid_path: Path,
        source_tier: str,
        target_tier: str,
        asr_pipeline: transformers.Pipeline,
        sampling_rate: int = 16000,
    ) -> "TextGridContainer":
        """Create a TextGrid with ASR predictions for each interval in a source tier.

        Reads an existing TextGrid, extracts audio segments corresponding to each
        non-empty interval in the source tier, runs ASR on each segment, and adds
        the predictions to a new target tier. The original tiers are preserved.

        Args:
            audio_in: Path to the audio file.
            textgrid_path: Path to the existing TextGrid file.
            source_tier: Name of the tier containing intervals to process.
            target_tier: Name for the new tier containing ASR predictions.
            asr_pipeline: Hugging Face transformers Pipeline for ASR.
            sampling_rate: Sampling rate for the audio

        Returns:
            A new TextGridContainer with all original tiers plus the new target tier.

        Raises:
            TypeError: If audio_in or textgrid_path is None.

        Note:
            If ASR fails for an interval, the error message is added to that interval
            in the target tier with the format "[Error]: {error_message}".
        """
        if audio_in is None:
            raise TypeError("Missing audio file")
        if textgrid_path is None:
            raise TypeError("Missing TextGrid input file")

        source_tg = tgt.io3.read_textgrid(textgrid_path, include_empty_intervals=True)
        tier = source_tg.get_tier_by_name(source_tier)
        ipa_tier = tgt.core.IntervalTier(name=target_tier)

        for interval in tier.intervals:
            start, end = interval.start_time, interval.end_time
            try:
                y, sr = librosa.load(audio_in, sr=sampling_rate, offset=start, duration=end - start)
                prediction = asr_pipeline(y)["text"]
                ipa_tier.add_annotation(tgt.core.Interval(start, end, prediction))

            except Exception as e:
                ipa_tier.add_annotation(tgt.core.Interval(start, end, f"[Error]: {e}"))

        source_tg.add_tier(ipa_tier)

        return cls(text_grid=source_tg)
