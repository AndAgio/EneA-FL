#!/usr/bin/env bash

cd ../data/raw_data

if [ ! -f nbaiot.csv ]; then
    fileid="16V6RlrhxI7r8WIJWSbSiK5AAdwL1G5qb"
    filename="nbaiot.csv"
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
    rm -rf cookie
fi

cd ../../preprocess
