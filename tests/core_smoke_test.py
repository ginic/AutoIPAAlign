#!/usr/bin/env python3
"""
Smoke test for autoipaalign-core package.

This test verifies that:
1. The package can be imported
2. Key modules are accessible
3. The CLI module and command classes can be imported
4. The CLI main function runs correctly

Used in CI to test built wheel and source distributions.
"""

import sys
from io import StringIO


def test_import_package():
    """Test that the autoipaalign_core package can be imported."""
    import autoipaalign_core


def test_import_key_modules():
    """Test that key modules can be imported."""
    from autoipaalign_core.textgrid_io import TextGridContainer


def test_import_cli():
    """Test that the CLI module can be imported."""
    from autoipaalign_core import cli


def test_import_cli_commands():
    """Test that CLI command classes can be imported."""
    from autoipaalign_core.cli import Transcribe, TranscribeIntervals


def test_cli_main_callable():
    """Test that the CLI main function runs and shows expected error."""
    from autoipaalign_core.cli import main

    # Capture stderr
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    try:
        main()
        raise AssertionError("main() should have raised SystemExit")
    except SystemExit as e:
        # Restore stderr and get captured output
        stderr_output = sys.stderr.getvalue()
        sys.stderr = old_stderr

        # Check that it exited with non-zero code (error)
        if e.code == 0:
            raise AssertionError("Expected non-zero exit code")

        # Check for expected error message in either stdout or stderr
        expected_text = "The following arguments are required: {transcribe,transcribe-intervals}"
        if expected_text not in stderr_output:
            raise AssertionError(f"Expected error message not found. Output: {stderr_output}")
    finally:
        # Ensure stdout/stderr are always restored
        sys.stderr = old_stderr


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running autoipaalign-core smoke tests")
    print("=" * 60)

    tests = [
        ("Testing package import", test_import_package),
        ("Testing key module imports", test_import_key_modules),
        ("Testing CLI module import", test_import_cli),
        ("Testing CLI command classes import", test_import_cli_commands),
        ("Testing CLI main function runs correctly", test_cli_main_callable),
    ]

    for description, test_func in tests:
        print(f"{description}...", end=" ")
        try:
            test_func()
            print("✓ SUCCESS")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            print("\n" + "=" * 60)
            print("SMOKE TEST FAILED")
            print("=" * 60)
            sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
