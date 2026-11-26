"""CLI entrypoint for the dartboard hit annotator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# Allow running directly as `python dart_annotator/main.py` as well as `python -m dart_annotator`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dart_annotator.app import DartAnnotatorApp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate dartboard images with 4-point homography and dart hit positions."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["images"],
        help="Image file(s) or directory(ies) containing dartboard images (default: images).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="annotations.json",
        help="Path to the output JSON file (default: annotations.json).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = DartAnnotatorApp(inputs=args.inputs, output_json=args.output)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
