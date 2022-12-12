#!/bin/bash

if [ $# -eq 0 ]; then
  echo "No device ID provided"
  exit 1
fi

/users/kszenes/ParTI/ParTI-own/C++/build/tests/test_tucker --dev $1 /users/kszenes/ParTI/ParTI-own/C++/example_tensors/sparse_5_5_5.tns 2 2 2 0 1 2