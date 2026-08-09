#!/bin/sh
set -eu

. /etc/default/muster
run="$MUSTER_ACTIVE_RUN"
source_directory="/srv/muster-archive/$run/replays"
published_root="/srv/muster/runs/$run"
published_directory="$published_root/replays"
published_link="/srv/muster/replays"
published_replay="/srv/muster/index.html"
temporary_replay=""
temporary_manifest=""
temporary_link=""
trap 'test -z "$temporary_replay" || rm -f "$temporary_replay"; test -z "$temporary_manifest" || rm -f "$temporary_manifest"; test -z "$temporary_link" || rm -f "$temporary_link"' EXIT HUP INT TERM

mkdir -p "$published_directory"
source_files=$(find "$source_directory" -maxdepth 1 -type f -name 'update-*.html' -printf '%f\n' | sort -V)

for replay_name in $source_files; do
    case "$replay_name" in
        update-[0-9]*.html) ;;
        *) continue ;;
    esac
    published_history="$published_directory/$replay_name"
    if ! test -f "$published_history"; then
        temporary_replay=$(mktemp "$published_directory/.replay.XXXXXX")
        cp "$source_directory/$replay_name" "$temporary_replay"
        test "$(wc -c < "$temporary_replay")" -gt 100000
        grep -q 'const replay=' "$temporary_replay"
        grep -q '"territory_i8":' "$temporary_replay"
        /usr/local/bin/muster-retemplate-replay /root/muster/muster/viewer/template.html "$temporary_replay"
        chmod 0644 "$temporary_replay"
        mv -f "$temporary_replay" "$published_history"
        temporary_replay=""
    fi
done

published_files=$(find "$published_directory" -maxdepth 1 -type f -name 'update-*.html' -printf '%f\n' | sort -V)
latest_name=""
for replay_name in $published_files; do latest_name="$replay_name"; done
test -n "$latest_name"
latest_update=${latest_name#update-}
latest_update=${latest_update%.html}
temporary_manifest=$(mktemp "$published_directory/.manifest.XXXXXX")
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
} > "$temporary_manifest"
chmod 0644 "$temporary_manifest"
mv -f "$temporary_manifest" "$published_directory/manifest.json"
temporary_manifest=""

current_target=$(readlink "$published_link" 2>/dev/null || true)
desired_target="runs/$run/replays"
if test "$current_target" != "$desired_target"; then
    temporary_link="/srv/muster/.replays-link.$$"
    ln -s "$desired_target" "$temporary_link"
    mv -Tf "$temporary_link" "$published_link"
    temporary_link=""
fi

if ! cmp -s "$published_directory/$latest_name" "$published_replay"; then
    temporary_replay=$(mktemp /srv/muster/.index.html.XXXXXX)
    cp "$published_directory/$latest_name" "$temporary_replay"
    chmod 0644 "$temporary_replay"
    mv -f "$temporary_replay" "$published_replay"
    temporary_replay=""
    logger -t muster-replay-sync "published Muster replay update $latest_update"
fi
