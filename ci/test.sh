#!/usr/bin/env bash
#
# Build and run the lecture test suites.
#
# Each lecture's test suite is cumulative: it covers everything implemented up to and including
# that lecture. All suites belonging to the same part of the course are therefore run against the
# same solution tree, which grows lecture by lecture:
#
#   L01 - L05 (part I)  -> solutions/part1
#   L06 - L10 (part II) -> solutions/part2
#
# Running every suite against the current solution is what catches regressions across lectures: if
# the work added in a later lecture breaks something an earlier lecture built, that lecture's suite
# fails. This holds as long as the lectures stay backward compatible. A suite that is deliberately
# left behind at an older lecture can opt out by setting 'FROZEN := 1' in its makefile.
#
# Parts whose solution hasn't been written yet are skipped, so this script is a no-op until at
# least one solution tree is in place.
#
# Usage:
#   ci/test.sh
set -euo pipefail

# Root directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directory holding the QAcademy Test framework.
FRAMEWORK_DIR="$ROOT_DIR/libs/test"

# Last lecture of part I. Lectures after this one belong to part II.
PART_ONE_LAST_LECTURE=5

# Number of test suites that were run.
suite_count=0

# Number of test suites that were skipped.
skip_count=0

################################################################################
# Print the solution directory belonging to the given lecture.
# Globals:
#   ROOT_DIR
#   PART_ONE_LAST_LECTURE
# Arguments:
#   $1 - Lecture number without leading zeros, e.g. 2 for L02.
################################################################################
solution_dir_for_lecture() {
    local lecture="$1"

    if [[ "$lecture" -le "$PART_ONE_LAST_LECTURE" ]]
    then
        echo "$ROOT_DIR/solutions/part1"
    else
        echo "$ROOT_DIR/solutions/part2"
    fi
}

################################################################################
# Print the lecture number the given test directory belongs to, without leading
# zeros, e.g. 2 for lectures/L02/appendix/exercises/test.
# Globals:
#   None
# Arguments:
#   $1 - Path to a lecture test directory.
################################################################################
lecture_of_test_dir() {
    local lecture
    lecture="$(basename "$(dirname "$(dirname "$(dirname "$1")")")")"
    lecture="${lecture#L}"
    echo "$((10#$lecture))"
}

# Terminate the script if the test framework submodule hasn't been checked out.
if [[ ! -f "$FRAMEWORK_DIR/Makefile" ]]
then
    echo "error: test framework not found in libs/test. Run 'git submodule update --init'." >&2
    exit 1
fi

# Build and run every test suite that has a solution to run against.
while IFS= read -r -d '' makefile
do
    test_dir="$(dirname "$makefile")"
    solution_dir="$(solution_dir_for_lecture "$(lecture_of_test_dir "$test_dir")")"

    # Skip suites that are deliberately left behind at an older lecture.
    if grep -q '^FROZEN := 1' "$makefile"
    then
        echo "Skipping $test_dir (frozen)"
        skip_count=$((skip_count + 1))
        continue
    fi

    # Skip parts whose solution hasn't been written yet.
    if [[ ! -d "$solution_dir/source" ]]
    then
        echo "Skipping $test_dir (no solution in $solution_dir)"
        skip_count=$((skip_count + 1))
        continue
    fi

    echo "Testing $test_dir against $solution_dir"
    make -C "$test_dir" clean build run ML_DIR="$solution_dir" QACADEMY_TEST_DIR="$FRAMEWORK_DIR"
    make -C "$test_dir" clean
    suite_count=$((suite_count + 1))
done < <(find "$ROOT_DIR/lectures" -path "*/appendix/exercises/test/Makefile" -print0 | sort -z)

echo "Ran $suite_count test suite(s), skipped $skip_count."
