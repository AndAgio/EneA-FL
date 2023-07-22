#!/bin/bash

# Loop 1000 times
for ((i=1; i<=100; i++))
do
    # Get the timestamp in milliseconds and append to "timestamps" file
    date +%s%3N >> timestamps

    # Run the Docker command and get the timestamp in milliseconds, then append to "timestamps" file
    sudo docker run --rm -it --network host \
      -v /home/jetson/Documents/test_container_instantiation:/enea \
      --workdir /enea \
      nvcr.io/nvidia/l4t-ml:r32.7.1-py3 \
      date +%s%3N >> timestamps
done







