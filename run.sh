#!/bin/bash

if [ -z "$1" ]; then
	echo "Usage: ./run.sh [image path]"
	exit 1
fi

if [ ! -f "$1" ]; then
	echo "Sorry, the given file doesn't exist."
	exit 1
fi

cd src
img="$1"
python final.py "../$img"