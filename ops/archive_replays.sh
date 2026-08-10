#!/bin/sh
set -eu

. /etc/default/muster
run="$MUSTER_ACTIVE_RUN"

source_directory="/home/ubuntu/muster-local-hex/runs/$run/replays"
archive_directory="/srv/muster-archive/$run/replays"
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

# Cluster runs (FASRC Cannon over the VPN). Best-effort: a down tunnel or
# login node must not fail the AWS archive above.
cannon_host=bowen@login.rc.fas.harvard.edu
cannon_root=/n/holylabs/avanderburg_lab/Lab/bowen/muster-runs
cannon_ssh() {
    ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$cannon_host" "$@"
}
for run in ${MUSTER_CANNON_RUNS:-}; do
    source_directory="$cannon_root/$run/replays"
    archive_directory="/srv/muster-archive/$run/replays"
    mkdir -p "$archive_directory"
    remote_files=$(cannon_ssh "find '$source_directory' -maxdepth 1 -type f -name 'update-*.html' -printf '%f\\n' 2>/dev/null" | sort -V) || continue
    for replay_name in $remote_files; do
        update=${replay_name#update-}
        update=${update%.html}
        case "$update" in
            ''|*[!0-9]*) continue ;;
        esac
        archived_replay="$archive_directory/$replay_name"
        if ! test -f "$archived_replay"; then
            temporary_replay=$(mktemp "$archive_directory/.replay.XXXXXX")
            if ! scp -q -o BatchMode=yes -o ConnectTimeout=5 \
                "$cannon_host:$source_directory/$replay_name" "$temporary_replay"; then
                rm -f "$temporary_replay"; temporary_replay=; continue
            fi
            test "$(wc -c < "$temporary_replay")" -gt 100000 || { rm -f "$temporary_replay"; temporary_replay=; continue; }
            grep -q 'const replay=' "$temporary_replay" || { rm -f "$temporary_replay"; temporary_replay=; continue; }
            remote_hash=$(cannon_ssh "sha256sum -- '$source_directory/$replay_name'" | awk '{print $1}')
            local_hash=$(sha256sum "$temporary_replay" | awk '{print $1}')
            test "$local_hash" = "$remote_hash" || { rm -f "$temporary_replay"; temporary_replay=; continue; }
            chmod 0644 "$temporary_replay"
            mv "$temporary_replay" "$archived_replay"
            temporary_replay=
        fi
        cannon_ssh "rm -f -- '$source_directory/$replay_name'" || true
    done
done
