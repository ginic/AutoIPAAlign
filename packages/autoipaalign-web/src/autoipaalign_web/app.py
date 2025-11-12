# Imports
from pathlib import Path
import tempfile

import gradio as gr

from autoipaalign_core.textgrid_io import TextGridContainer, write_textgrids_to_target
from autoipaalign_core.speech_recognition import ASRPipeline

# Constants
TEXTGRID_DIR = tempfile.mkdtemp()
DEFAULT_MODEL = "ginic/full_dataset_train_3_wav2vec2-large-xlsr-53-buckeye-ipa"
TEXTGRID_DOWNLOAD_TEXT = "Download TextGrid file"
TEXTGRID_NAME_INPUT_LABEL = "TextGrid file name"

TITLE = "AutoIPA: Automated IPA transcription"

INTRO_BLOCK = f"""# {TITLE}
Experiment with producing phonetic transcriptions of uploaded or recorded audio using Wav2Vec2.0-based automatic
speech recognition (ASR) models!

The AutoIPA project is a collaboration between Virginia Partridge of the UMass Center for Data Science and Artificial
Intelligence and Joe Pater of UMass Linguistics. Its goal is to make automated IPA transcription more useful
to linguists (and others!).
Our first step was to fine-tune a Wav2Vec 2.0 model on the Buckeye corpus, which you can try out here.
Our next steps will be to extend our work to other varieties of English and other languages.
Please reach out to us if you have any questions or comments about our work or have related work to share!
More details are on our [project website](https://websites.umass.edu/comphon/autoipa-automated-ipa-transcription/).

If you use our software, please cite our AMP paper:
Partridge, Virginia, Joe Pater, Parth Bhangla, Ali Nirheche and Brandon Prickett. 2025/to appear. [AI-assisted analysis of phonological variation in English](https://docs.google.com/presentation/d/1IJrfokvX5T_fKkiFXmcYEgRI2ZRwgFU4zU1tNC-iYl0/edit?usp=sharing). Special session on Deep Phonology, AMP 2025, UC Berkeley. To appear in the Proceedings of AMP 2025.
"""
UMASS_MAROON = gr.themes.Color(
    c50="#f8e8eb",
    c100="#eec6cc",
    c200="#d39092",
    c300="#c06769",
    c400="#c44849",
    c500="#c43732",
    c600="#b63030",
    c700="#a5282b",
    c800="#982325",
    c900="#811c1c",
    c950="#811c1c",
)
THEME = gr.themes.Default(primary_hue=UMASS_MAROON)


# Selection of models
VALID_MODELS = [
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
    "excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k",
    "excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified",
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
                "asr_pipeline": ASRPipeline(model_name=model_name),
                "model_name": model_name,
            }

        prediction = model_state["asr_pipeline"].predict(audio_in)
        return prediction, model_state
    except Exception as e:
        raise gr.Error(f"Failed to load model: {str(e)}")


def get_textgrid_contents(audio_in, textgrid_tier_name, transcription_prediction):
    if audio_in is None or transcription_prediction is None:
        return ""

    tg_container = TextGridContainer.from_audio_and_transcription(
        audio_in, textgrid_tier_name, transcription_prediction
    )
    return tg_container.export_to_long_textgrid_str()


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


def transcribe_intervals(model_name, audio_in, textgrid_path, source_tier, target_tier, model_state):
    if audio_in is None or textgrid_path is None:
        return "Missing audio or TextGrid input file.", model_state

    # Check if correct model is loaded, reload if necessary
    if model_state["model_name"] != model_name:
        model_state = {
            "asr_pipeline": ASRPipeline(model_name=model_name),
            "model_name": model_name,
        }

    # Reuse the ASRPipeline from model_state (efficient if model unchanged)
    asr_pipeline = model_state["asr_pipeline"]

    tg_container = TextGridContainer.from_textgrid_with_predict_intervals(
        audio_in, Path(textgrid_path.name), source_tier, target_tier, asr_pipeline
    )

    return tg_container.export_to_long_textgrid_str(), model_state


def extract_tier_names(textgrid_file):
    try:
        tg_container = TextGridContainer.from_textgrid_file(Path(textgrid_file.name))
        tier_names = tg_container.get_tier_names()
        return gr.update(choices=tier_names, value=tier_names[0] if tier_names else None)
    except Exception:
        return gr.update(choices=[], value=None)


def validate_textgrid_for_intervals(audio_path, textgrid_file):
    try:
        if not audio_path or not textgrid_file:
            return gr.update(interactive=False)

        tg_container = TextGridContainer.from_textgrid_file(Path(textgrid_file.name))
        tg_container.validate_against_audio_duration(audio_path)
        return gr.update(interactive=True)

    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Invalid TextGrid or audio file:\n{str(e)}")


def transcribe_multiple_files(model_name, audio_files, model_state, tier_name):
    try:
        if not audio_files:
            return [], None, model_state

        # Check if correct model is loaded, reload if necessary
        if model_state["model_name"] != model_name:
            model_state = {
                "asr_pipeline": ASRPipeline(model_name=model_name),
                "model_name": model_name,
            }

        table_data = []
        text_grids = []
        audio_paths = []

        for file in audio_files:
            # Use ASRPipeline for prediction (consistent with CLI)
            prediction = model_state["asr_pipeline"].predict(file)

            # Use TextGridContainer to create TextGrid (modular, consistent with CLI)
            tg_container = TextGridContainer.from_audio_and_transcription(file, tier_name, prediction)

            table_data.append([Path(file).name, prediction])
            text_grids.append(tg_container)
            audio_paths.append(Path(file))

        # Write all TextGrids to zip using CLI function
        zip_path = Path(tempfile.mkdtemp()) / "textgrids.zip"
        write_textgrids_to_target(audio_paths, text_grids, zip_path, is_zip=True, is_overwrite=True)

        return table_data, str(zip_path), model_state

    except Exception as e:
        raise gr.Error(f"Transcription failed: {str(e)}")


def launch_demo():
    initial_model = {
        "asr_pipeline": ASRPipeline(model_name=DEFAULT_MODEL),
        "model_name": DEFAULT_MODEL,
    }

    with gr.Blocks(title=TITLE, theme=THEME) as demo:
        gr.Markdown(INTRO_BLOCK)

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
            inputs=[model_name, interval_audio, interval_textgrid_file, tier_names, target_tier, model_state],
            outputs=[interval_result, model_state],
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
