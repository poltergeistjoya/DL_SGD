#!/bin/bash

set -e

# podman pull tensorflow/tensorflow:2.9.1 # for CPU tf
podman pull tensorflow/tensorflow:2.9.1-gpu # for GPU tf
podman buildx build --platform linux/amd64 -t tf .
podman run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "flake8 ."
podman run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "black ."
podman run -u $(id -u):$(id -g) -t -v=${PWD}:/app tf "find -iname *.py  -print | parallel 'a2ps {} -o {}.ps  && ps2pdf {}.ps {}.pdf'"
podman run -u $(id -u):$(id -g) -ti -v=${PWD}:/app tf "${*}"

# Usage:
# $ ./tf.sh <command string>

# Example invocation:
# $ ./tf.sh python linear-example/main.py


# $ ./tf.sh  gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=linear_example.pdf linear-example/main.py.pdf linear-example/fit.pdf
