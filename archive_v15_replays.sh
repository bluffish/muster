#!/bin/sh
set -eu

source_directory=/home/ubuntu/muster-local-hex/runs/local-hex-v15-memory-v1/replays
archive_directory=/srv/muster-archive/local-hex-v15-memory-v1/replays
remote=ubuntu@52.90.113.210
identity=/root/.ssh/id_ed25519
instance=i-0257496d5842c9d83
availability_zone=us-east-1b
temporary_replay=
trap 'test -z "$temporary_replay" || rm -f "$temporary_replay"' EXIT HUP INT TERM

remote_ssh() {
    ssh -q -i "$identity" -o BatchMode=yes -o IdentitiesOnly=yes \
        -o ConnectTimeout=3 "$remote" "$@"
}

ensure_remote() {
    if remote_ssh true 2>/dev/null; then
        return
    fi
    /usr/local/bin/aws ec2-instance-connect send-ssh-public-key \
        --instance-id "$instance" \
        --availability-zone "$availability_zone" \
        --instance-os-user ubuntu \
        --ssh-public-key "file://$identity.pub" \
        --region us-east-1 >/dev/null
}

mkdir -p "$archive_directory"
ensure_remote
remote_files=$(remote_ssh "find '$source_directory' -maxdepth 1 -type f -name 'update-*.html' -printf '%f\\n'" | sort -V)

for replay_name in $remote_files; do
    update=${replay_name#update-}
    update=${update%.html}
    case "$update" in
        ''|*[!0-9]*) continue ;;
    esac

    archived_replay="$archive_directory/$replay_name"
    if ! test -f "$archived_replay"; then
        temporary_replay=$(mktemp "$archive_directory/.replay.XXXXXX")
        scp -q -i "$identity" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=3 \
            "$remote:$source_directory/$replay_name" "$temporary_replay"
        test "$(wc -c < "$temporary_replay")" -gt 100000
        grep -q 'const replay=' "$temporary_replay"
        grep -q '"territory_i8":' "$temporary_replay"
        remote_hash=$(remote_ssh "sha256sum -- '$source_directory/$replay_name'" | awk '{print $1}')
        local_hash=$(sha256sum "$temporary_replay" | awk '{print $1}')
        test "$local_hash" = "$remote_hash"
        chmod 0644 "$temporary_replay"
        mv "$temporary_replay" "$archived_replay"
        temporary_replay=
    fi
    remote_ssh "rm -f -- '$source_directory/$replay_name'"
done
