#!/bin/bash

MODULES=(
    "/data/CODE/python/adopy_tests"
    "/data/Dropbox/RDATA/myR"
)

for path in "${MODULES[@]}"; do
    echo "==> Indexing $path..."
    (cd "$path" && codegraph index)
done

echo "Done."
