from __future__ import annotations

import argparse
from pathlib import Path

from pysideband import __version__, __description__, __software_panel_message__
from pysideband.config import load_config
from pysideband.launcher import LaunchConfig, launch
from pysideband.workflow import run_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__description__,
        epilog=__software_panel_message__,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("input_file", type=Path, help="Input file for the calculation.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_file: Path = args.input_file

    project_config = load_config(input_file)

    launch_config = LaunchConfig.from_config(
        project_config, [str(input_file.resolve())]
    )

    def run() -> int:
        run_workflow(project_config)
        return 0

    return launch(run, launch_config)
