# Imports
from pathlib import Path
import tempfile
import os
import gradio as gr
import librosa
import tgt.core
import tgt.io3
import soundfile as sf
import zipfile
from transformers import pipeline

# Constants
TEXTGRID_DIR = tempfile.mkdtemp()
DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"
TEXTGRID_DOWNLOAD_TEXT = "Download TextGrid file"
TEXTGRID_NAME_INPUT_LABEL = "TextGrid file name"

# Selection of models
VALID_MODELS = [
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
    "ginic/full_dataset_train_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/full_dataset_train_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/full_dataset_train_4_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/full_dataset_train_5_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/data_seed_bs64_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/data_seed_bs64_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/data_seed_bs64_3_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/data_seed_bs64_4_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_30_female_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_30_female_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_30_female_3_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_30_female_4_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_30_female_5_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_70_female_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_70_female_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_70_female_3_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_70_female_4_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/gender_split_70_female_5_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_old_only_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_old_only_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_old_only_3_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_young_only_1_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_young_only_2_wav2vec2-large-xlsr-53-buckeye-ipa",
    "ginic/vary_individuals_young_only_3_wav2vec2-large-xlsr-53-buckeye-ipa",
]


def load_model_and_predict(
    model_name: str,
    audio_in: str,
    model_state: dict,
):
    try:
        if audio_in is None:
            return (
                "",
                model_state,
                gr.Textbox(label=TEXTGRID_NAME_INPUT_LABEL, interactive=False),
            )

        if model_state["model_name"] != model_name:
            model_state = {
                "loaded_model": pipeline(task="automatic-speech-recognition", model=model_name),
                "model_name": model_name,
            }

        prediction = model_state["loaded_model"](audio_in)["text"]
        return prediction, model_state
    except Exception as e:
        raise gr.Error(f"Failed to load model: {str(e)}")


# TODO replace with the TextGridContainer.from_audio_and_transcription
def get_textgrid_contents(audio_in, textgrid_tier_name, transcription_prediction):
    if audio_in is None or transcription_prediction is None:
        return ""

    duration = librosa.get_duration(path=audio_in)

    annotation = tgt.core.Interval(0, duration, transcription_prediction)
    transcription_tier = tgt.core.IntervalTier(start_time=0, end_time=duration, name=textgrid_tier_name)
    transcription_tier.add_annotation(annotation)
    textgrid = tgt.core.TextGrid()
    textgrid.add_tier(transcription_tier)
    return tgt.io3.export_to_long_textgrid(textgrid)


# TODO replace with TextGridContainer.write_textgrid()
def write_textgrid(textgrid_contents, textgrid_filename):
    """Writes the text grid contents to a named file in the temporary directory.
    Returns the path for download.
    """
    textgrid_path = Path(TEXTGRID_DIR) / Path(textgrid_filename).name
    textgrid_path.write_text(textgrid_contents)
    return textgrid_path


def get_interactive_download_button(textgrid_contents, textgrid_filename):
    return gr.DownloadButton(
        label=TEXTGRID_DOWNLOAD_TEXT,
        variant="primary",
        interactive=True,
        value=write_textgrid(textgrid_contents, textgrid_filename),
    )


# TODO Replace with the TextGridContainer.from_textgrid_with_predicted_intervals function
def transcribe_intervals(audio_in, textgrid_path, source_tier, target_tier, model_state):
    if audio_in is None or textgrid_path is None:
        return "Missing audio or TextGrid input file."

    tg = tgt.io.read_textgrid(textgrid_path.name)
    tier = tg.get_tier_by_name(source_tier)
    ipa_tier = tgt.core.IntervalTier(name=target_tier)

    for interval in tier.intervals:
        if not interval.text.strip():  # Skip empty text intervals
            continue

        start, end = interval.start_time, interval.end_time
        try:
            y, sr = librosa.load(audio_in, sr=None, offset=start, duration=end - start)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                sf.write(temp_audio.name, y, sr)
                prediction = model_state["loaded_model"](temp_audio.name)["text"]
                ipa_tier.add_annotation(tgt.core.Interval(start, end, prediction))
                os.remove(temp_audio.name)
        except Exception as e:
            ipa_tier.add_annotation(tgt.core.Interval(start, end, f"[Error]: {str(e)}"))

    tg.add_tier(ipa_tier)
    tgt_str = tgt.io3.export_to_long_textgrid(tg)

    return tgt_str


# TODO replace with TextGridContainer.get_tier_names function as much as possible
def extract_tier_names(textgrid_file):
    try:
        tg = tgt.io.read_textgrid(textgrid_file.name)
        tier_names = [tier.name for tier in tg.tiers]
        return gr.update(choices=tier_names, value=tier_names[0] if tier_names else None)
    except Exception as e:
        return gr.update(choices=[], value=None)


# TODO replace with TextGridContainer.validate_against_audio_duration
def validate_textgrid_for_intervals(audio_path, textgrid_file):
    try:
        if not audio_path or not textgrid_file:
            return gr.update(interactive=False)

        audio_duration = librosa.get_duration(path=audio_path)
        tg = tgt.io.read_textgrid(textgrid_file.name)
        tg_end_time = max(tier.end_time for tier in tg.tiers)

        # TextGrid ends later than audio
        if tg_end_time > audio_duration:
            raise gr.Error(
                f"TextGrid ends at {tg_end_time:.2f}s but audio is only {audio_duration:.2f}s. "
                "Please upload matching files."
            )

        epsilon = 0.01
        if abs(tg_end_time - audio_duration) > epsilon:
            gr.Warning(
                f"TextGrid ends at {tg_end_time:.2f}s but audio is {audio_duration:.2f}s. "
                "Only the annotated portion will be transcribed."
            )

        return gr.update(interactive=True)

    except Exception as e:
        raise gr.Error(f"Invalid TextGrid or audio file:\n{str(e)}")


def transcribe_multiple_files(model_name, audio_files, model_state, tier_name):
    try:
        if not audio_files:
            return [], None, model_state

        if model_state["model_name"] != model_name:
            model_state = {
                "loaded_model": pipeline(task="automatic-speech-recognition", model=model_name),
                "model_name": model_name,
            }

        table_data = []
        tg_paths = []

        for file in audio_files:
            prediction = model_state["loaded_model"](file)["text"]
            duration = librosa.get_duration(path=file)

            annotation = tgt.core.Interval(0, duration, prediction)
            transcription_tier = tgt.core.IntervalTier(0, duration, tier_name)
            transcription_tier.add_annotation(annotation)

            tg = tgt.core.TextGrid()
            tg.add_tier(transcription_tier)

            tg_str = tgt.io3.export_to_long_textgrid(tg)
            tg_filename = Path(file).with_suffix(".TextGrid").name
            tg_path = Path(TEXTGRID_DIR) / tg_filename
            tg_path.write_text(tg_str)

            table_data.append([Path(file).name, prediction])
            tg_paths.append(tg_path)

        # ZIP generation
        zip_path = Path(tempfile.mkdtemp()) / "textgrids.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for tg in tg_paths:
                zipf.write(tg, arcname=tg.name)

        return table_data, str(zip_path), model_state

    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")


def launch_demo():
    initial_model = {
        "loaded_model": pipeline(task="automatic-speech-recognition", model=DEFAULT_MODEL),
        "model_name": DEFAULT_MODEL,
    }

    with gr.Blocks() as demo:
        gr.Markdown("""# Automatic International Phonetic Alphabet Transcription
This demo allows you to experiment with producing phonetic transcriptions of uploaded or recorded audio using a selected automatic speech recognition (ASR) model.\n
If you're unsure which model to use, the default `ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa` should give good performance.""")

        # Dropdown for model selection
        model_name = gr.Dropdown(
            VALID_MODELS,
            value=DEFAULT_MODEL,
            label="IPA transcription ASR model",
            info="Select the model to use for prediction.",
        )

        # Dropdown for transcription type selection
        transcription_type = gr.Dropdown(
            choices=["Full Audio", "Multiple Full Audio", "TextGrid Interval"],
            label="Transcription Type",
            value=None,
            interactive=True,
        )

        model_state = gr.State(value=initial_model)

        # Full audio transcription section
        with gr.Column(visible=False) as full_audio_section:
            full_audio = gr.Audio(type="filepath", show_download_button=True, label="Upload Audio File")
            full_transcribe_btn = gr.Button("Transcribe Full Audio", interactive=False, variant="primary")
            full_prediction = gr.Textbox(label="IPA Transcription", show_copy_button=True)

            full_textgrid_tier = gr.Textbox(label="TextGrid Tier Name", value="IPA", interactive=True)

            full_textgrid_contents = gr.Textbox(label="TextGrid Contents", show_copy_button=True)
            full_download_btn = gr.DownloadButton(label=TEXTGRID_DOWNLOAD_TEXT, interactive=False, variant="primary")
            full_reset_btn = gr.Button("Reset", variant="secondary")

        # Multiple full audio transcription section
        with gr.Column(visible=False) as multiple_full_audio_section:
            multiple_full_audio = gr.File(file_types=[".wav"], label="Upload Audio File(s)", file_count="multiple")
            multiple_full_textgrid_tier = gr.Textbox(label="TextGrid Tier Name", value="IPA")
            multiple_full_transcribe_btn = gr.Button("Transcribe Audio Files", interactive=False, variant="primary")

            multiple_full_table = gr.Dataframe(
                headers=["Filename", "Transcription"],
                interactive=False,
                label="IPA Transcriptions",
                datatype=["str", "str"],
            )

            multiple_full_zip_download_btn = gr.File(label="Download All as ZIP", interactive=False)
            multiple_full_reset_btn = gr.Button("Reset", variant="secondary")

        # Interval transcription section
        with gr.Column(visible=False) as interval_section:
            interval_audio = gr.Audio(type="filepath", show_download_button=True, label="Upload Audio File")
            interval_textgrid_file = gr.File(file_types=["text", ".TextGrid"], label="Upload TextGrid File")
            tier_names = gr.Dropdown(label="Source Tier (existing)", choices=[], interactive=True)
            target_tier = gr.Textbox(label="Target Tier (new)", value="IPATier", placeholder="e.g. IPATier")

            interval_transcribe_btn = gr.Button("Transcribe Intervals", interactive=False, variant="primary")
            interval_result = gr.Textbox(label="IPA Interval Transcription", show_copy_button=True, interactive=False)
            interval_download_btn = gr.DownloadButton(
                label=TEXTGRID_DOWNLOAD_TEXT, interactive=False, variant="primary"
            )
            interval_reset_btn = gr.Button("Reset", variant="secondary")

        # Section visibility toggle
        transcription_type.change(
            fn=lambda t: (
                gr.update(visible=t == "Full Audio"),
                gr.update(visible=t == "Multiple Full Audio"),
                gr.update(visible=t == "TextGrid Interval"),
            ),
            inputs=transcription_type,
            outputs=[full_audio_section, multiple_full_audio_section, interval_section],
        )

        # Enable full transcribe button after audio uploaded
        full_audio.change(
            fn=lambda audio: gr.update(interactive=audio is not None),
            inputs=full_audio,
            outputs=full_transcribe_btn,
        )

        # Full transcription logic
        full_transcribe_btn.click(
            fn=load_model_and_predict,
            inputs=[model_name, full_audio, model_state],
            outputs=[full_prediction, model_state],
        )

        full_prediction.change(
            fn=get_textgrid_contents,
            inputs=[full_audio, full_textgrid_tier, full_prediction],
            outputs=[full_textgrid_contents],
        )

        full_textgrid_contents.change(
            fn=lambda tg_text, audio_path: get_interactive_download_button(
                tg_text, Path(audio_path).with_suffix(".TextGrid").name if audio_path else "output.TextGrid"
            ),
            inputs=[full_textgrid_contents, full_audio],
            outputs=[full_download_btn],
        )

        full_reset_btn.click(
            fn=lambda: (None, "", "", "", gr.update(interactive=False)),
            outputs=[full_audio, full_prediction, full_textgrid_contents, full_download_btn],
        )

        # Enable interval transcribe button only when both files are uploaded
        interval_audio.change(
            fn=validate_textgrid_for_intervals,
            inputs=[interval_audio, interval_textgrid_file],
            outputs=[interval_transcribe_btn],
        )

        interval_textgrid_file.change(
            fn=validate_textgrid_for_intervals,
            inputs=[interval_audio, interval_textgrid_file],
            outputs=[interval_transcribe_btn],
        )

        # Interval logic
        interval_textgrid_file.change(
            fn=extract_tier_names,
            inputs=[interval_textgrid_file],
            outputs=[tier_names],
        )

        interval_transcribe_btn.click(
            fn=transcribe_intervals,
            inputs=[interval_audio, interval_textgrid_file, tier_names, target_tier, model_state],
            outputs=[interval_result],
        )

        interval_result.change(
            fn=lambda tg_text, audio_path: gr.update(
                value=write_textgrid(tg_text, Path(audio_path).with_suffix("").name + "_IPA.TextGrid"),
                interactive=True,
            ),
            inputs=[interval_result, interval_audio],
            outputs=[interval_download_btn],
        )

        interval_reset_btn.click(
            fn=lambda: (None, None, gr.update(choices=[]), "IPATier", "", gr.update(interactive=False)),
            outputs=[
                interval_audio,
                interval_textgrid_file,
                tier_names,
                target_tier,
                interval_result,
                interval_download_btn,
            ],
        )

        # Multiple full audio transcription logic
        multiple_full_audio.change(
            fn=lambda files: gr.update(interactive=bool(files)),
            inputs=multiple_full_audio,
            outputs=multiple_full_transcribe_btn,
        )

        multiple_full_transcribe_btn.click(
            fn=transcribe_multiple_files,
            inputs=[model_name, multiple_full_audio, model_state, multiple_full_textgrid_tier],
            outputs=[multiple_full_table, multiple_full_zip_download_btn, model_state],
        )

        multiple_full_reset_btn.click(
            fn=lambda: (None, "", [], None, gr.update(interactive=False)),
            outputs=[
                multiple_full_audio,
                multiple_full_textgrid_tier,
                multiple_full_table,
                multiple_full_zip_download_btn,
                multiple_full_transcribe_btn,
            ],
        )

    demo.launch(max_file_size="100mb")


if __name__ == "__main__":
    launch_demo()
