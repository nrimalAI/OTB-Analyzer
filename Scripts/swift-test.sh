#!/bin/bash
# Runs the ChessVision package's tests: ./Scripts/swift-test.sh [swift test arguments]
#
# With Xcode selected this is plain `swift test`. With only the Command Line
# Tools (the Mac mini, where Xcode's licence is not accepted) it adds the
# flags that let `swift test` find Swift Testing there: the tests use Swift
# Testing, not XCTest, because the Command Line Tools ship no XCTest.
set -euo pipefail
cd "$(dirname "$0")/../Packages/ChessVision"

developer_dir="${DEVELOPER_DIR:-$(xcode-select -p)}"
flags=()
if [[ "$developer_dir" == */CommandLineTools ]]; then
    frameworks="$developer_dir/Library/Developer/Frameworks"
    if [[ ! -d "$frameworks/Testing.framework" ]]; then
        echo "No Testing.framework in $frameworks: update the Command Line Tools." >&2
        exit 1
    fi
    flags=(
        -Xswiftc -F -Xswiftc "$frameworks"
        -Xlinker -F -Xlinker "$frameworks"
        -Xlinker -rpath -Xlinker "$frameworks"
        # The Command Line Tools' _Testing_Foundation overlay has no module.
        -Xswiftc -Xfrontend -Xswiftc -disable-cross-import-overlays
    )
fi
exec swift test ${flags[@]+"${flags[@]}"} "$@"
