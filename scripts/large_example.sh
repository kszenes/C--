#!/bin/bash

if [ $# -eq 0 ]; then
  echo "No device ID provided"
  exit 1
fi

/users/kszenes/ParTI/ParTI-own/C++/build/tests/test_ttm --mode 0 --dev $1 /users/kszenes/ParTI/ParTI-own/C++/example_tensors/sparse_2000_22.tns /users/kszenes/ParTI/ParTI-own/C++/example_tensors/dense_32_2000.tns
