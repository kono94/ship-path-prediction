#!/bin/bash

echo "sourcing into virtualenv; setting PYTHONPATH"
source .venv/bin/activate
export PYTHONPATH=$(pwd)