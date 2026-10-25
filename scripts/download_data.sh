#!/usr/bin/env bash
# Download NPLIB1 (the CANOPUS training export released with MIST) into data/canopus and verify the
# split and labels files.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d data/canopus/spec_files ]; then
  mkdir -p data
  wget -O data/canopus_train_export.tar "https://zenodo.org/record/8316682/files/canopus_train_export_v2.tar"
  tar -xf data/canopus_train_export.tar -C data
  mv data/canopus_train_export data/canopus
  rm -f data/canopus_train_export.tar
fi

sha256sum -c - <<'SUMS'
c83a602d2079f88e91407f873f7f3a50cc5fbb0d535d255668fde915cd1d374f  data/canopus/splits/canopus_hplus_100_0.tsv
cac36f5ba028684958b9712691ed254ccfa4d1a72b83f14b63020a4089eb950e  data/canopus/labels.tsv
SUMS
echo "NPLIB1 ready in data/canopus"
