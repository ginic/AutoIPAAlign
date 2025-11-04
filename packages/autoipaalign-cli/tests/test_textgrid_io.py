"""Unit tests for textgrid_io module"""

from pathlib import Path

import pytest
import tgt.core
import tgt.io3

from autoipaalign_cli.textgrid_io import TextGridContainer


@pytest.fixture
def sample_textgrid():
    """Create a sample TextGrid for testing"""
    tg = tgt.core.TextGrid()
    tier = tgt.core.IntervalTier(start_time=0, end_time=5.0, name="words")
    tier.add_annotation(tgt.core.Interval(0, 2.5, "hello"))
    tier.add_annotation(tgt.core.Interval(2.5, 5.0, "world"))
    tg.add_tier(tier)
    return tg


@pytest.fixture
def temp_textgrid_file(sample_textgrid, tmp_path):
    """Create a temporary TextGrid file for testing"""
    textgrid_path = tmp_path / "test.TextGrid"
    textgrid_str = tgt.io3.export_to_long_textgrid(sample_textgrid)
    textgrid_path.write_text(textgrid_str)
    return textgrid_path


def test_export_to_long_textgrid_str(sample_textgrid):
    """Test exporting TextGrid to long format string"""
    container = TextGridContainer(text_grid=sample_textgrid)
    result = container.export_to_long_textgrid_str()
    assert isinstance(result, str)
    assert "File type = \"ooTextFile\"" in result


def test_get_tier_names(sample_textgrid):
    """Test retrieving tier names from TextGrid"""
    container = TextGridContainer(text_grid=sample_textgrid)
    tier_names = container.get_tier_names()
    assert tier_names == ["words"]


def test_write_textgrid(sample_textgrid, tmp_path):
    """Test writing TextGrid to file"""
    container = TextGridContainer(text_grid=sample_textgrid)
    filename = Path("test.TextGrid")
    result_path = container.write_textgrid(tmp_path, filename)

    assert result_path.exists()
    assert result_path.name == "test.TextGrid"
    assert result_path.parent == tmp_path


def test_from_textgrid_file(temp_textgrid_file):
    """Test creating TextGridContainer from file"""
    result = TextGridContainer.from_textgrid_file(temp_textgrid_file)

    assert isinstance(result, TextGridContainer)
    assert len(result.text_grid.tiers) == 1
    assert result.text_grid.tiers[0].name == "words"
    assert len(result.text_grid.tiers[0].intervals) == 2


def test_from_audio_and_transcription(mocker):
    """Test creating TextGrid from audio and transcription"""
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.get_duration", return_value=3.5)

    result = TextGridContainer.from_audio_and_transcription(
        audio_in="/path/to/audio.wav",
        textgrid_tier_name="transcription",
        transcription="hello world"
    )

    assert isinstance(result, TextGridContainer)
    assert len(result.text_grid.tiers) == 1
    assert result.text_grid.tiers[0].name == "transcription"
    assert result.text_grid.tiers[0].intervals[0].text == "hello world"


def test_from_textgrid_with_predicted_intervals(mocker, temp_textgrid_file):
    """Test creating TextGrid with ASR predictions"""
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.load", return_value=([0.1, 0.2, 0.3], 16000))
    mocker.patch("autoipaalign_cli.textgrid_io.sf.write")
    mocker.patch("autoipaalign_cli.textgrid_io.os.path.exists", return_value=True)
    mock_remove = mocker.patch("autoipaalign_cli.textgrid_io.os.remove")

    mock_pipeline = mocker.Mock()
    mock_pipeline.return_value = {"text": "həˈloʊ"}

    result = TextGridContainer.from_textgrid_with_predicted_intervals(
        audio_in="/path/to/audio.wav",
        textgrid_path=temp_textgrid_file,
        source_tier="words",
        target_tier="ipa",
        asr_pipeline=mock_pipeline
    )

    assert isinstance(result, TextGridContainer)
    assert len(result.text_grid.tiers) == 2
    assert result.text_grid.get_tier_by_name("ipa").intervals[0].text == "həˈloʊ"
    mock_remove.assert_called()
