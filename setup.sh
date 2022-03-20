#!/bin/bash

echo "sourcing into virtualenv; setting PYTHONPATH"
source env/bin/activate
export PYTHONPATH=$(pwd)
echo $(python --version)