"""Utilities for manipulating TextGrid files with audio and transcriptions.

This module provides a container class for working with Praat TextGrid files,
including reading from and writing to files, creating TextGrids from audio and
transcriptions, and generating new tiers using automatic speech recognition (ASR).
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import librosa
import soundfile as sf
import tgt.core
import tgt.io3
import transformers


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

    def write_textgrid(self, directory: Path, filename: Path) -> Path:
        """Write the TextGrid to a file in the specified directory.

        Args:
            directory: The directory where the TextGrid file will be written.
            filename: The desired filename (only the name portion will be used).

        Returns:
            The full path to the written TextGrid file.
        """
        textgrid_path = Path(directory) / Path(filename).name
        textgrid_path.write_text(self.export_to_long_textgrid_str())
        return textgrid_path

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

        duration = librosa.get_duration(path=audio_in)

        annotation = tgt.core.Interval(0, duration, transcription)
        transcription_tier = tgt.core.IntervalTier(start_time=0, end_time=duration, name=textgrid_tier_name)
        transcription_tier.add_annotation(annotation)
        textgrid = tgt.core.TextGrid()
        textgrid.add_tier(transcription_tier)
        return cls(text_grid=textgrid)

    @classmethod
    def from_textgrid_with_predicted_intervals(
        cls,
        audio_in: str | os.PathLike[str],
        textgrid_path: Path,
        source_tier: str,
        target_tier: str,
        asr_pipeline: transformers.Pipeline,
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
            raise TypeError("Missing TextGrid input file.")

        tg = tgt.io3.read_textgrid(textgrid_path)
        tier = tg.get_tier_by_name(source_tier)
        ipa_tier = tgt.core.IntervalTier(name=target_tier)

        for interval in tier.intervals:
            if not interval.text.strip():  # Skip empty text intervals
                continue

            start, end = interval.start_time, interval.end_time
            try:
                y, sr = librosa.load(audio_in, sr=None, offset=start, duration=end - start)
                temp_audio_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                        temp_audio_path = temp_audio.name
                        sf.write(temp_audio_path, y, sr)

                    prediction = asr_pipeline(temp_audio_path)["text"]
                    ipa_tier.add_annotation(tgt.core.Interval(start, end, prediction))
                finally:
                    if temp_audio_path and os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            except Exception as e:
                ipa_tier.add_annotation(tgt.core.Interval(start, end, f"[Error]: {str(e)}"))

        tg.add_tier(ipa_tier)

        return cls(text_grid=tg)
