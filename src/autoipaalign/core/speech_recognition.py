"""Automatic Speech Recognition model support for predicting transcriptions from audio files.

Note that we are using the librosa library for processing individual audio files,
rather than the HuggingFace datasets[audio] for now, to decrease installation
complexity. Per https://huggingface.co/docs/datasets/audio_load, the Datasets
audio processing relies on ffmpeg, which is an external library that may be
more difficult to install. However, doing so limits our batch data processing
options.

# TODO: Add batch audio processing via an optional datasets[audio]
"""

from dataclasses import dataclass, field
import logging
import os

import librosa
import transformers

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"


@dataclass
class TranscriptionChunk:
    """Represents a single aligned transcribed interval (word/phone) with timestamp information."""

    text: str
    """The text transcribed within the interval"""

    timestamp: tuple[float, float]
    """Start and end time in seconds."""


@dataclass
class TranscriptionWithTimestamps:
    """Represents a full transcription with transcribed sub-intervals and their timestamps."""

    text: str
    """The full transcription text."""

    chunks: list[TranscriptionChunk]
    """List of individual characters/phones with their timestamps."""


def load_audio(
    audio_path: str | os.PathLike[str],
    sampling_rate: int,
    interval: tuple[float, float] | None = None,
):
    """Load audio file with optional interval extraction.

    Args:
        audio_path: Path to the audio file
        sampling_rate: Sampling rate for audio preprocessing
        interval: Optional tuple of (start, end) times in seconds

    Returns:
        Audio array loaded at the specified sampling rate
    """
    if interval:
        logger.debug("Loading interval %s from audio %s", interval, audio_path)
        start, end = interval
        y, sr = librosa.load(audio_path, sr=sampling_rate, offset=start, duration=end - start)
    else:
        logger.debug("Loading audio %s", audio_path)
        y, sr = librosa.load(audio_path, sr=sampling_rate)
    return y


@dataclass
class ASRPipeline:
    """Handles loading and configuration of the Transformer pipeline"""

    model_name: str = field(default=DEFAULT_MODEL)
    """The name of the HuggingFace model used to transcribe speech."""

    device: int | str = field(default=-1)
    """Index of the device for model inference. Defaults to -1 for CPU."""

    sampling_rate: int = field(default=16000, kw_only=True)
    """Sampling rate for audio preprocessing. Defaults to 16K."""

    _model_pipe: transformers.Pipeline = field(init=False)

    def __post_init__(self):
        logger.info("Loading model: %s", self.model_name)
        self._model_pipe = transformers.pipeline(
            "automatic-speech-recognition", model=self.model_name, device=self.device
        )

    def predict(
        self,
        audio_path: str | os.PathLike[str],
        interval: tuple[float, float] | None = None,
    ) -> str:
        """Predict transcription for an audio file.

        Args:
            audio_path: Path to the audio file
            interval: Optional tuple of (start, end) times in seconds

        Returns:
            Transcription text
        """
        y = load_audio(audio_path, self.sampling_rate, interval)
        logger.debug("Predicting transcription for %s with model %s", audio_path, self.model_name)
        transcription = self._model_pipe(y)["text"]
        return transcription

    def predict_with_timestamps(
        self,
        audio_path: str | os.PathLike[str],
        interval: tuple[float, float] | None = None,
    ) -> TranscriptionWithTimestamps:
        """Predict transcription with character-level timestamps for an audio file.

        Args:
            audio_path: Path to the audio file
            interval: Optional tuple of (start, end) times in seconds

        Returns:
            TranscriptionWithTimestamps containing full text and character-level chunks
        """
        y = load_audio(audio_path, self.sampling_rate, interval)
        logger.debug(
            "Predicting transcription with timestamps for %s with model %s",
            audio_path,
            self.model_name,
        )
        result = self._model_pipe(y, return_timestamps="char")

        # Collect TranscriptionChunk objects
        chunks = []
        for c in result.get("chunks", []):
            # chunk timestamp as np.float
            timestamp = c["timestamp"]
            chunk = TranscriptionChunk(text=c["text"], timestamp=(timestamp[0].item(), timestamp[1].item()))
            chunks.append(chunk)

        return TranscriptionWithTimestamps(text=result["text"], chunks=chunks)
