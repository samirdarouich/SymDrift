#!/usr/bin/env bash

# This script will sync all files from your local machine to the server,
# keeping your file layout the same relative to your home directory.

# Use -d or --delete flag to enable deletion of remote files that don't exist locally

base_dir=$(dirname $(readlink -f $0))
server_dir="tspath"
node=vinhtong@129.69.197.70

# Initialize delete_flag and create array for files
delete_flag=""
files=()

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--delete)
            delete_flag="--delete"
            shift
            ;;
        -n|--node)
            if [[ $# -gt 1 ]]; then
                node="$2"
                shift 2
            else
                echo "Error: --node requires an argument" >&2
                exit 1
            fi
            ;;
        *)
            files+=("$1")
            shift
            ;;
    esac
done

echo "Syncing ${base_dir} to ${server_dir}"

set -x
if [ ${#files[@]} -eq 0 ]; then
    # No specific files provided, sync everything
    rsync -rhv --update ${delete_flag} --safe-links \
        --include=configs/untracked \
        --exclude=scripts/notebooks \
        --exclude=untracked/ \
        --exclude-from=.gitignore \
        --exclude-from=.git/info/exclude \
        ${base_dir}/ \
        ${node}:${server_dir}/
else
    # Handle individual file transfers
    for file in "${files[@]}"; do
        scp "${file}" "${node}:${server_dir}/$(dirname ${file})"
    done
fi
