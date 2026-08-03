#!/usr/bin/env bash
#
# Build the lecture demo code.
#
# Usage:
#   ci/build.sh
set -euo pipefail

# Root directory.
ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")/.."

# Build every lecture demo and solution tree that has a Makefile. Test directories are pruned:
# their makefiles need a solution to build against, and are handled by ci/test.sh instead.
for root in "$ROOT_DIR/lectures" "$ROOT_DIR/solutions"
do
    # Skip the solution trees when none have been written yet.
    if [[ ! -d "$root" ]]
    then
        continue
    fi

    while IFS= read -r -d '' makefile; do
        demo_dir="$(dirname "$makefile")"
        echo "Building $demo_dir"
        make -C "$demo_dir" build
    done < <(find "$root" -type d -name test -prune -o -name Makefile -print0 | sort -z)
done
