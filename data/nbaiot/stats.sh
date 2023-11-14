#!/usr/bin/env bash

NAME="nbaiot"

cd ../utils

python3 stats.py --name $NAME

cd ../$NAME
