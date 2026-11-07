#!/bin/sh
# Execute NEO's already-separated argv without evaluating task text.
set -eu
if [ "$#" -eq 0 ]; then
    echo "usage: neo-launch-macos.sh PROGRAM [ARG ...]" >&2
    exit 64
fi
exec "$@"
