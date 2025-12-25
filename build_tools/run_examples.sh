#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert \
  --to notebook \
  --inplace \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3"

for notebook in docs/notebooks/*.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook"
done
