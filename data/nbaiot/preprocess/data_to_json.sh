#!/usr/bin/env bash

if [ ! -d "../data" ]; then
  mkdir ../data
fi
if [ ! -d "../data/raw_data" ]; then
  mkdir ../data/raw_data
fi
if [ ! -f ../data/raw_data/nbaiot.csv ]; then
  echo "------------------------------"
  echo "retrieving raw data"
  
  ./get_data.sh
  echo "finished retrieving raw data"
fi

if [ ! -d "../data/all_data" ]; then
  echo "------------------------------"
  echo "converting data to .json format"
  mkdir ../data/all_data
  python3 data_to_json.py
  echo "finished converting data to .json format"
fi

