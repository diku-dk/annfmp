#!/bin/bash
python setup.py clean
find annfmp -name "*.so" -delete
find annfmp -name "*.pyc" -delete
find annfmp -name "__pycache__" -type d -exec rm -rf {} +
find annfmp -name "*_wrap.c" -delete
find annfmp -name "wrapper_*" -delete
rm -rf annfmp.egg-info
python -m pip install -e . --no-build-isolation