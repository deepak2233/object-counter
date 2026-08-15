"""Command line entrypoint: same use cases, no HTTP.

Useful for batch runs and for checking a model in a terminal without starting a
server. It is a second driver on the same ports, which is the point of the
architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from counter.bootstrap import build_services
from counter.config import Settings
from counter.domain.errors import ObjectCounterError
from counter.domain.models import Image
from counter.observability.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="object-counter", description=__doc__)
    parser.add_argument("command", choices=["detect", "count", "models"])
    parser.add_argument("image", nargs="?", type=Path, help="path to an image file")
    parser.add_argument("--threshold", type=float, default=None, help="confidence cut-off, 0..1")
    parser.add_argument("--model", dest="model_name", default=None, help="model name to use")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level, "text")
    services = build_services(settings)

    try:
        if args.command == "models":
            payload = {
                "default_model": getattr(services.registry, "default_model", ""),
                "models": [asdict(info) for info in services.registry.available()],
            }
        else:
            if args.image is None:
                print("an image path is required", file=sys.stderr)
                return 2

            image = Image(content=args.image.read_bytes(), filename=args.image.name)
            threshold = settings.default_threshold if args.threshold is None else args.threshold
            action = (
                services.detect_action(args.model_name)
                if args.command == "detect"
                else services.count_action(args.model_name)
            )
            payload = asdict(action.execute(image, threshold))

        print(json.dumps(payload, indent=2, default=str))
        return 0
    except ObjectCounterError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"could not read image: {exc}", file=sys.stderr)
        return 1
    finally:
        services.close()


if __name__ == "__main__":
    raise SystemExit(main())
