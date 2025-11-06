"""Unit tests for textgrid_io module"""

from pathlib import Path
import zipfile

import pytest
import tgt.core
import tgt.io3

from autoipaalign_cli.textgrid_io import TextGridContainer, write_textgrids_to_target


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
    assert result == tgt.io3.export_to_long_textgrid(sample_textgrid)


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
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.get_duration", return_value=5.5)

    result = TextGridContainer.from_audio_and_transcription(
        audio_in="/path/to/audio.wav", textgrid_tier_name="transcription", transcription="hello world"
    )

    assert isinstance(result, TextGridContainer)
    assert len(result.text_grid.tiers) == 1
    assert result.text_grid.start_time == 0
    assert result.text_grid.end_time == 5.5
    assert len(result.get_tier_names()) == 1

    # Check that the correct tier was added
    test_tier = result.text_grid.tiers[0]
    assert test_tier.name == "transcription"
    assert len(test_tier.intervals) == 1
    assert test_tier.intervals[0].text == "hello world"
    assert test_tier.intervals[0].start_time == 0
    assert test_tier.intervals[0].end_time == 5.5


def test_from_textgrid_with_predict_intervals(mocker, temp_textgrid_file):
    """Test creating TextGrid with mock ASR predictions"""
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.load", return_value=([0.1, 0.2, 0.3], 16000))

    mock_pipeline = mocker.Mock()
    mock_pipeline.return_value = {"text": "həloʊ"}

    result = TextGridContainer.from_textgrid_with_predict_intervals(
        audio_in="/path/to/audio.wav",
        textgrid_path=temp_textgrid_file,
        source_tier="words",
        target_tier="ipa",
        asr_pipeline=mock_pipeline,
    )

    # Check that word tier is still present
    assert isinstance(result, TextGridContainer)
    assert len(result.text_grid.tiers) == 2
    tier_names = result.get_tier_names()
    assert set(tier_names) == set(["words", "ipa"])

    # The same mocked prediction was added at different intervals
    interval1 = result.text_grid.get_tier_by_name("ipa").intervals[0]
    assert interval1.text == "həloʊ"
    assert interval1.start_time == 0
    assert interval1.end_time == 2.5

    interval2 = result.text_grid.get_tier_by_name("ipa").intervals[1]
    assert interval2.text == "həloʊ"
    assert interval2.start_time == 2.5
    assert interval2.end_time == 5.0

    # Prediction and temp file removal both happened twice
    assert mock_pipeline.call_count == 2


def test_from_audio_with_predict_transcription(mocker):
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.load", return_value=([0.1, 0.2, 0.3], 16000))
    mocker.patch("autoipaalign_cli.textgrid_io.librosa.get_duration", return_value=5.5)
    mock_pipeline = mocker.Mock()
    mock_pipeline.return_value = {"text": "hello"}
    tg = TextGridContainer.from_audio_with_predict_transcription(
        audio_in="/path/to/audio.wav", textgrid_tier_name="transcription", asr_pipeline=mock_pipeline
    )

    assert tg.get_tier_names() == ["transcription"]
    tier = tg.text_grid.get_tier_by_name("transcription")
    assert len(tier.intervals) == 1
    interval = tier.intervals[0]
    assert interval.text == "hello"
    assert interval.start_time == 0
    assert interval.end_time == 5.5


def test_write_textgrids_to_directory(sample_textgrid, tmp_path):
    """Test writing TextGrids to directory"""
    audio_paths = [Path("test1.wav"), Path("test2.wav")]
    text_grids = [TextGridContainer(sample_textgrid), TextGridContainer(sample_textgrid)]
    target = tmp_path / "output"

    write_textgrids_to_target(audio_paths, text_grids, target, is_zip=False)

    assert target.is_dir()
    assert (target / "test1.TextGrid").exists()
    assert (target / "test2.TextGrid").exists()


def test_write_textgrids_to_zip(sample_textgrid, tmp_path):
    """Test writing TextGrids to zip file"""
    audio_paths = [Path("test1.wav"), Path("test2.wav")]
    text_grids = [TextGridContainer(sample_textgrid), TextGridContainer(sample_textgrid)]
    target = tmp_path / "output.zip"

    write_textgrids_to_target(audio_paths, text_grids, target, is_zip=True)

    assert target.exists()
    with zipfile.ZipFile(target, "r") as zipf:
        names = zipf.namelist()
        assert "test1.TextGrid" in names
        assert "test2.TextGrid" in names


def test_write_textgrids_to_target_no_overwrite(sample_textgrid, tmp_path):
    """Test that write_textgrids_to_target raises error when is_overwrite=False and file exists"""
    audio_paths = [Path("test.wav")]
    text_grids = [TextGridContainer(sample_textgrid)]
    target = tmp_path / "output.zip"

    # Write first time
    write_textgrids_to_target(audio_paths, text_grids, target, is_zip=True)

    # Should raise error on second write with is_overwrite=False
    with pytest.raises(OSError, match="already exists"):
        write_textgrids_to_target(audio_paths, text_grids, target, is_zip=True, is_overwrite=False)
