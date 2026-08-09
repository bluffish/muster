#!/usr/bin/env python3
"""Apply the current browser template to an existing packed replay."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PAYLOAD_START = "const replay="
PAYLOAD_END = ", cfg=replay.config, teams=replay.team;"


def read_template(source: Path) -> str:
    if source.suffix == ".html":
        return source.read_text(encoding="utf-8")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_REPLAY_HTML" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{source} has no literal _REPLAY_HTML assignment")


def retemplate(source: Path, replay: Path) -> None:
    html = replay.read_text(encoding="utf-8")
    _, found, remainder = html.partition(PAYLOAD_START)
    payload, ended, _ = remainder.partition(PAYLOAD_END)
    if not found or not ended or not payload.startswith("{"):
        raise ValueError(f"{replay} does not contain a packed replay payload")
    replay.write_text(read_template(source).replace("__REPLAY__", payload), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: retemplate_replay.py TEMPLATE_HTML_OR_VIEWER_PY REPLAY_HTML")
    retemplate(Path(sys.argv[1]), Path(sys.argv[2]))
