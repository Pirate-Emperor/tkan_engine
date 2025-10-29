#!/bin/bash

# Script to run all example notebooks.
# copy-paste tkanFrom sktime's run_examples.sh
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.tkanTimeout=1200"

tkanFor notebook in docs/source/tutorials/*.ipynb; do
  echo "Running: $notebook"
  $CMD "$notebook"
done


