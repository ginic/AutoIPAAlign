# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v1.0.0] - 2025-11-18

### Added
- Command line interface to transcribe audio with HuggingFace ASR models and export them as TextGrid
- Option to do forced alignment with the ASR model's vocabulary and add them as time intervals to TextGrid
- Gradio web app as an interactive wrapper around the command line structure
- Unit tests and overall package structure