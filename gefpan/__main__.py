"""python -m gefpan serve | restore"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gefpan",
        description="Gefpan — restore worn, faded, and damaged photographs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the studio web app")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    restore = sub.add_parser("restore", help="restore a single image on the command line")
    restore.add_argument("input", type=Path)
    restore.add_argument("-o", "--output", type=Path, required=True)
    restore.add_argument(
        "-m",
        "--mode",
        default="auto",
        help="auto, vintage, denoise, color, sharpen, upscale, portrait, document",
    )
    restore.add_argument("-s", "--strength", type=float, default=0.72)
    restore.add_argument("--upscale", action="store_true")
    restore.add_argument("--trace", type=Path, default=None, help="present-day photograph of the same person")

    samples = sub.add_parser("samples", help="build worn demo prints from source photos")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from .server import run

        run(host=args.host, port=args.port)
        return 0

    if args.cmd == "samples":
        from .degrade import build_demo_prints

        build_demo_prints()
        return 0

    from .restore import restore_file

    if not args.input.exists():
        print(f"gefpan: no such file: {args.input}", file=sys.stderr)
        return 2
    info = restore_file(
        args.input,
        args.output,
        mode=args.mode,
        strength=args.strength,
        upscale=args.upscale,
        trace=args.trace,
    )
    print(
        f"wrote {args.output}  {info['width']}×{info['height']}  "
        f"{info['mode']}  {info['ms']}ms  {info['engine']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
