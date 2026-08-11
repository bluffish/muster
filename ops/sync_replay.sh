#!/bin/sh
set -eu

. /etc/default/muster
active_run="$MUSTER_ACTIVE_RUN"
archive_root="/srv/muster-archive"
published_root="/srv/muster"
published_link="$published_root/replays"
template="/root/muster/muster/viewer/template.html"
temporary_file=""
trap 'test -z "$temporary_file" || rm -f "$temporary_file"' EXIT HUP INT TERM

# Publish every archived run under /srv/muster/runs/<run>/replays with a
# per-run manifest; replay pages use links relative to their manifest, so
# each directory is self-contained.
for source_directory in "$archive_root"/*/replays; do
    test -d "$source_directory" || continue
    run=$(basename "$(dirname "$source_directory")")
    published_directory="$published_root/runs/$run/replays"
    mkdir -p "$published_directory"

    for replay_path in $(find "$source_directory" -maxdepth 1 -type f -name 'update-*.html' | sort -V); do
        replay_name=$(basename "$replay_path")
        case "$replay_name" in
            update-[0-9]*.html) ;;
            *) continue ;;
        esac
        published_history="$published_directory/$replay_name"
        if ! test -f "$published_history"; then
            temporary_file=$(mktemp "$published_directory/.replay.XXXXXX")
            cp "$replay_path" "$temporary_file"
            test "$(wc -c < "$temporary_file")" -gt 100000
            grep -q 'const replay=' "$temporary_file"
            grep -q '"territory_i8":' "$temporary_file"
            /usr/local/bin/muster-retemplate-replay "$template" "$temporary_file"
            chmod 0644 "$temporary_file"
            mv -f "$temporary_file" "$published_history"
            temporary_file=""
        fi
        # The published copy carries the full payload (retemplating extracts
        # it from any replay file), so the archive copy is redundant once a
        # valid published copy exists. The archive is a transit area only.
        if test -f "$published_history" \
            && test "$(wc -c < "$published_history")" -gt 100000 \
            && grep -q 'const replay=' "$published_history"; then
            rm -f "$replay_path"
        fi
    done

    published_files=$(find "$published_directory" -maxdepth 1 -type f -name 'update-*.html' -printf '%f\n' | sort -V)
    test -n "$published_files" || continue
    latest_name=""
    for replay_name in $published_files; do latest_name="$replay_name"; done
    latest_update=${latest_name#update-}
    latest_update=${latest_update%.html}
    temporary_file=$(mktemp "$published_directory/.manifest.XXXXXX")
    {
        printf '{"run":"%s","current":%s,"updates":[' "$run" "$latest_update"
        separator=""
        for replay_name in $published_files; do
            update=${replay_name#update-}
            update=${update%.html}
            printf '%s%s' "$separator" "$update"
            separator=,
        done
        printf ']}\n'
    } > "$temporary_file"
    chmod 0644 "$temporary_file"
    mv -f "$temporary_file" "$published_directory/manifest.json"
    temporary_file=""
done

# The bare domain serves the active run: /replays points at it and the
# root page redirects to its newest replay.
current_target=$(readlink "$published_link" 2>/dev/null || true)
desired_target="runs/$active_run/replays"
# Only follow the active run once it has published content; until then the
# previous run keeps serving, so a freshly-started run never breaks the site.
if test -f "$published_root/$desired_target/manifest.json" \
    && test "$current_target" != "$desired_target"; then
    temporary_file="$published_root/.replays-link.$$"
    ln -s "$desired_target" "$temporary_file"
    mv -Tf "$temporary_file" "$published_link"
    temporary_file=""
fi

active_manifest="$published_root/$desired_target/manifest.json"
if test -f "$active_manifest"; then
    active_current=$(sed -n 's/.*"current":\([0-9]*\).*/\1/p' "$active_manifest")
    temporary_file=$(mktemp "$published_root/.index.XXXXXX")
    cat > "$temporary_file" <<INDEX
<!doctype html><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=/replays/update-$active_current.html">
<title>Muster</title>
<body style="background:#0e1116;color:#8b98a5;font:14px ui-monospace,monospace;padding:40px">
<a style="color:#35a7ff" href="/replays/update-$active_current.html">latest replay ($active_run · update $active_current)</a>
 · <a style="color:#8b98a5" href="/runs.html">all runs</a>
INDEX
    chmod 0644 "$temporary_file"
    mv -f "$temporary_file" "$published_root/index.html"
    temporary_file=""
fi

# Machine-readable run list for the charts page's run switcher.
temporary_file=$(mktemp "$published_root/.runsindex.XXXXXX")
{
    printf '['
    separator=""
    for manifest in $(ls -t "$published_root"/runs/*/replays/manifest.json 2>/dev/null); do
        run=$(sed -n 's/.*"run":"\([^"]*\)".*/\1/p' "$manifest")
        printf '%s"%s"' "$separator" "$run"
        separator=,
    done
    printf ']\n'
} > "$temporary_file"
chmod 0644 "$temporary_file"
mv -f "$temporary_file" "$published_root/runs/index.json"
temporary_file=""

# Dashboard: one row per published run, newest activity first.
temporary_file=$(mktemp "$published_root/.runs.XXXXXX")
{
    cat <<'HEAD'
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Muster - runs</title>
<body style="background:#0e1116;color:#c9d4df;font:14px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;padding:36px 20px">
<div style="max-width:640px;margin:0 auto">
<div style="font-weight:700;letter-spacing:.24em;color:#8b98a5;font-size:12px">MUSTER</div>
<h1 style="font-size:20px;margin:8px 0 20px">Training runs</h1>
<table style="border-collapse:collapse;width:100%">
<tr style="color:#8b98a5;text-align:left"><th style="padding:4px 12px 4px 0">run</th><th style="padding:4px 12px 4px 0">replays</th><th style="padding:4px 0">latest</th></tr>
HEAD
    for manifest in $(ls -t "$published_root"/runs/*/replays/manifest.json 2>/dev/null); do
        run=$(sed -n 's/.*"run":"\([^"]*\)".*/\1/p' "$manifest")
        current=$(sed -n 's/.*"current":\([0-9]*\).*/\1/p' "$manifest")
        count=$(grep -o ',' "$manifest" | wc -l)
        count=$((count + 1))
        marker=""
        case " $active_run ${MUSTER_CANNON_RUNS:-} " in
            *" $run "*) marker=' <span style="color:#ffc440">&#9679; live</span>' ;;
        esac
        printf '<tr><td style="padding:4px 12px 4px 0;border-top:1px solid #232c38">%s%s</td><td style="padding:4px 12px 4px 0;border-top:1px solid #232c38">%s</td><td style="padding:4px 0;border-top:1px solid #232c38"><a style="color:#35a7ff" href="/runs/%s/replays/update-%s.html">update %s</a> &#183; <a style="color:#8b98a5" href="/charts.html?run=%s">charts</a></td></tr>\n' \
            "$run" "$marker" "$count" "$run" "$current" "$current" "$run"
    done
    printf '</table></div>\n'
} > "$temporary_file"
chmod 0644 "$temporary_file"
mv -f "$temporary_file" "$published_root/runs.html"
temporary_file=""
