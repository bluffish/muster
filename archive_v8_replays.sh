#!/bin/sh
set -eu

source_directory=/home/ubuntu/muster-local-hex/runs/local-hex-v8-entity-attention-v1/replays
archive_directory=/srv/muster-archive/local-hex-v8-entity-attention-v1/replays
temporary_replay=
trap 'test -z "$temporary_replay" || rm -f "$temporary_replay"' EXIT HUP INT TERM

mkdir -p "$archive_directory"
remote_files=$(ssh -q bowen3 "find '$source_directory' -maxdepth 1 -type f -name 'update-*.html' -printf '%f\\n'" | sort -V)

for replay_name in $remote_files; do
    update=${replay_name#update-}
    update=${update%.html}
    case "$update" in
        ''|*[!0-9]*) continue ;;
    esac

    archived_replay="$archive_directory/$replay_name"
    if ! test -f "$archived_replay"; then
        temporary_replay=$(mktemp "$archive_directory/.replay.XXXXXX")
        scp -q "bowen3:$source_directory/$replay_name" "$temporary_replay"
        test "$(wc -c < "$temporary_replay")" -gt 100000
        grep -q 'const replay=' "$temporary_replay"
        grep -q '"territory_i8":' "$temporary_replay"
        remote_hash=$(ssh -q bowen3 "sha256sum -- '$source_directory/$replay_name'" | awk '{print $1}')
        local_hash=$(sha256sum "$temporary_replay" | awk '{print $1}')
        test "$local_hash" = "$remote_hash"
        chmod 0644 "$temporary_replay"
        mv "$temporary_replay" "$archived_replay"
        temporary_replay=
    fi
    ssh -q bowen3 "rm -f -- '$source_directory/$replay_name'"
done
