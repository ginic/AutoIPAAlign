"""Utilities for manipulating TextGrid files with audio and transcriptions"""

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
    text_grid: tgt.core.TextGrid

    def export_to_long_textgrid_str(self) -> str:
        return tgt.io3.export_to_long_textgrid(self.text_grid)

    def get_tier_names(self) -> list[str]:
        return [tier.name for tier in self.text_grid.tiers]

    def write_textgrid(self, directory: Path, filename: Path) -> Path:
        """Writes the TextGrid contents to a named file in the specified directory.
        Returns the path where the file was written to.
        """
        textgrid_path = Path(directory) / Path(filename).name
        textgrid_path.write_text(self.export_to_long_textgrid_str())
        return textgrid_path

    @classmethod
    def from_textgrid_file(cls, textgrid_file: Path):
        tg = tgt.io3.read_textgrid(textgrid_file)
        return cls(text_grid=tg)

    @classmethod
    def from_audio_and_transcription(
        cls, audio_in: str | os.PathLike[str], textgrid_tier_name: str, transcription: str
    ) -> "TextGridContainer":
        """Creates a TextGrid with a single tier containing the
        transcription across an interval of the full audio duration.

        Args:
            audio_in: Path to the audio file
            textgrid_tier_name: Desired name for the transcription's tier in the TextGrid
            transcription: Transcription to add as the value of the tier

        Returns:
            TextGridContainer
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
