#!/bin/bash

export CXX=clang
export CXXFLAGS="-Wall -Wextra -Werror \
    -Wno-c++98-compat \
    -Wno-exit-time-destructors \
    -fsized-deallocation \
    "
