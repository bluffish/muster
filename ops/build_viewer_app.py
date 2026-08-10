#!/usr/bin/env python3
"""Build view.html: the fetch-mode viewer app from the inline template.

The app renders any replay from compressed payloads hosted on the
Hugging Face dataset (bluffish/muster-replays), addressed as
/view.html?run=<run>&u=<update>. Manifests stay on the local server so
live runs keep updating; navigation goes through the app URL scheme.
"""

from __future__ import annotations

import sys
from pathlib import Path

DATA_BASE = "https://huggingface.co/datasets/bluffish/muster-replays/resolve/main"

LOADER = """const appQuery=new URLSearchParams(location.search);
const appRun=appQuery.get("run")||"";
const appUpdate=Number(appQuery.get("u"));
const appDataBase="__DATA_BASE__";
function pageUrl(update) { return "/view.html?run="+encodeURIComponent(appRun)+"&u="+update; }
async function loadReplay() {
  if (!appRun||!Number.isInteger(appUpdate)) throw new Error("missing ?run= and ?u= parameters");
  const response=await fetch(appDataBase+"/"+appRun+"/update-"+appUpdate+".json.gz");
  if (!response.ok) throw new Error("replay data not found ("+response.status+")");
  const stream=response.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}
let replay;
try {
  replay=await loadReplay();
} catch (error) {
  document.body.insertAdjacentHTML("beforeend",
    '<div style="position:fixed;inset:0;display:grid;place-items:center;background:#0e1116;color:#c9d4df;font:15px ui-monospace,monospace;z-index:99">'+
    "could not load replay: "+String(error && error.message || error)+"</div>");
  throw error;
}
const cfg=replay.config, teams=replay.team;"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    template = (root / "muster" / "viewer" / "template.html").read_text()

    payload_line = "const replay=__REPLAY__, cfg=replay.config, teams=replay.team;"
    assert template.count(payload_line) == 1
    app = template.replace(payload_line, LOADER.replace("__DATA_BASE__", DATA_BASE))

    old = "<script>"
    assert app.count(old) == 1
    app = app.replace(old, '<script type="module">')

    old = 'fetch("manifest.json",{cache:"no-store"})'
    assert app.count(old) == 1
    app = app.replace(
        old,
        'fetch("/runs/"+encodeURIComponent(appRun)+"/replays/manifest.json",{cache:"no-store"})',
    )

    old = 'location.replace("update-"+latestUpdate+".html")'
    assert app.count(old) == 1
    app = app.replace(old, "location.replace(pageUrl(latestUpdate))")

    old = 'location.replace("update-"+updates.value+".html")'
    assert app.count(old) == 1
    app = app.replace(old, "location.replace(pageUrl(Number(updates.value)))")

    output = root / "ops" / "view.html" if len(sys.argv) < 2 else Path(sys.argv[1])
    output.write_text(app)
    print("wrote", output, len(app), "bytes")


if __name__ == "__main__":
    main()
