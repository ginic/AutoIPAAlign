"""Unit tests for CLI module"""

import zipfile

import pytest
import tgt.io3

from autoipaalign_cli.cli import ASRPipeline, Transcribe, TranscribeIntervals


@pytest.fixture
def mock_asr_pipeline(mocker):
    """Create a mock ASR pipeline"""
    mock_pipeline = mocker.Mock(spec=ASRPipeline)
    mock_pipeline.model_name = "test-model"
    mock_pipeline._model_pipe = mocker.Mock()
    mock_pipeline._model_pipe.return_value = {"text": "test transcription"}
    return mock_pipeline


def test_transcribe_run_directory(mock_asr_pipeline, tmp_path, shared_datadir):
    """Test Transcribe.run() writing to directory"""
    # Full file transcription does all files at once

    audio_path = shared_datadir / "test1.wav"
    transcribe = Transcribe(
        asr=mock_asr_pipeline,
        audio_paths=[audio_path],
        output_target=tmp_path / "output",
        tier_name="ipa",
        zipped=False,
    )

    transcribe.run()

    assert (tmp_path / "output").is_dir()
    assert (tmp_path / "output" / "test1.TextGrid").exists()
    mock_asr_pipeline._model_pipe.assert_called_once()

    # Check internal contents
    tg = tgt.io3.read_textgrid(tmp_path / "output" / "test1.TextGrid")
    assert len(tg.tiers) == 1
    assert tg.tiers[0].name == "ipa"
    assert len(tg.tiers[0].intervals) == 1
    assert tg.tiers[0].intervals[0].text == "test transcription"
    # Full audio duration
    assert tg.tiers[0].intervals[0].start_time == 0
    assert tg.tiers[0].intervals[0].end_time == 2.2798125


def test_transcribe_run_zip(mock_asr_pipeline, tmp_path, shared_datadir):
    """Test Transcribe.run() writing to zip file"""
    # Full file transcription does all files at once
    audio_path = shared_datadir / "test1.wav"
    transcribe = Transcribe(
        asr=mock_asr_pipeline,
        audio_paths=[audio_path],
        output_target=tmp_path / "output.zip",
        tier_name="ipa",
        zipped=True,
    )

    transcribe.run()

    zip_path = tmp_path / "output.zip"
    assert zip_path.exists()

    extract_path = tmp_path / "extract"

    # Extract and check internal contents
    with zipfile.ZipFile(zip_path, "r") as zipf:
        assert "test1.TextGrid" in zipf.namelist()
        zipf.extractall(extract_path)

    tg = tgt.io3.read_textgrid(extract_path / "test1.TextGrid")
    assert len(tg.tiers) == 1
    assert tg.tiers[0].name == "ipa"
    assert tg.tiers[0].intervals[0].text == "test transcription"
    # Full audio duration
    assert tg.tiers[0].intervals[0].start_time == 0
    assert tg.tiers[0].intervals[0].end_time == 2.2798125


def test_transcribe_intervals_run(mock_asr_pipeline, tmp_path, shared_datadir):
    """Test TranscribeIntervals.run()"""
    # Intervals predict one at a time

    audio_path = shared_datadir / "test1.wav"
    textgrid_path = shared_datadir / "test1.TextGrid"

    transcribe_intervals = TranscribeIntervals(
        asr=mock_asr_pipeline,
        audio_path=audio_path,
        textgrid_path=textgrid_path,
        output_target=tmp_path,
        source_tier="words",
        target_tier="ipa",
    )

    transcribe_intervals.run()

    output_path = tmp_path / "test1.TextGrid"
    assert output_path.exists()

    assert mock_asr_pipeline._model_pipe.call_count == 11

    # Check internal contents
    tg = tgt.io3.read_textgrid(output_path)
    assert len(tg.tiers) == 2
    assert "words" in tg.get_tier_names()
    assert "ipa" in tg.get_tier_names()
    ipa_tier = tg.get_tier_by_name("ipa")
    # Matches interval durations in test1.TextGrid
    assert len(ipa_tier.intervals) == 11
    assert all(interval.text == "test transcription" for interval in ipa_tier.intervals)
    assert ipa_tier.start_time == 0
    assert ipa_tier.end_time == 2.273


def test_transcribe_intervals_run_error_handling(mock_asr_pipeline, tmp_path, mocker, shared_datadir):
    """Test TranscribeIntervals.run() handles ASR errors gracefully"""
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.load", side_effect=Exception("Load error"))
    # Intervals predict one at a time
    mock_asr_pipeline._model_pipe.return_value = {"text": "test transcription"}
    audio_path = shared_datadir / "test1.wav"
    textgrid_path = shared_datadir / "test1.TextGrid"

    transcribe_intervals = TranscribeIntervals(
        asr=mock_asr_pipeline,
        audio_path=audio_path,
        textgrid_path=textgrid_path,
        output_target=tmp_path,
        source_tier="words",
        target_tier="ipa",
    )

    # Should complete without raising, errors written to TextGrid
    transcribe_intervals.run()

    assert (tmp_path / "test1.TextGrid").exists()

    # Check internal contents show errors
    tg = tgt.io3.read_textgrid(tmp_path / "test1.TextGrid")
    ipa_tier = tg.get_tier_by_name("ipa")
    assert len(ipa_tier.intervals) == 11
    # Matches interval durations in test1.TextGrid
    assert ipa_tier.start_time == 0
    assert ipa_tier.end_time == 2.273
    assert all("[Error]" in interval.text for interval in ipa_tier.intervals)
