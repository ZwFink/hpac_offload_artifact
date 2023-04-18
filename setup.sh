#!/bin/bash

if [ -d "benchmarks/input_data" ]; then
  echo "Directory benchmarks/input_data already exists. Skipping download and unzip."
else
  # Download the archive using wget
  wget https://uofi.box.com/shared/static/7p8eb0ppw6zhtdr7krhiw9kat52lmwez.gz

  # Unzip the archive using tar
  tar -xzf 7p8eb0ppw6zhtdr7krhiw9kat52lmwez.gz

  # Create the 'benchmarks' directory if it doesn't exist
  mkdir -p benchmarks

  # Move the 'input_data' directory to the 'benchmarks' directory
  mv input_data benchmarks/
  rm 7p8eb0ppw6zhtdr7krhiw9kat52lmwez.gz
fi

