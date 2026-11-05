#!/bin/bash
# Build the TransNet feature exporter (modified Scallop2) into bin/transnet-exporter.
# Run inside the activated conda environment from environment.yml:
#   conda activate transnet && bash exporter/build.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "$here")"
prefix="${CONDA_PREFIX:?activate the transnet conda environment first}"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
compiler="${CXX:-c++}"
flags=(-std=c++17 -O2 -pthread -I"$here/src" -I"$here/lib/gtf" -I"$here/lib/graph" -I"$here/lib/util" -I"$prefix/include")
objects=()
for source in "$here"/lib/{gtf,graph,util}/*.cc "$here"/src/*.cc; do
    object="$build/$(basename "$(dirname "$source")")_$(basename "${source%.cc}").o"
    "$compiler" "${flags[@]}" -c "$source" -o "$object" &
    objects+=("$object")
    # limit parallel compiles
    while [ "$(jobs -r | wc -l)" -ge "${THREADS:-4}" ]; do sleep 0.2; done
done
wait
mkdir -p "$root/bin"
"$compiler" "${objects[@]}" -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lhts -pthread -o "$root/bin/transnet-exporter"
echo "Built $root/bin/transnet-exporter"
