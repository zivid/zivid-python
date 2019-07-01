#!/bin/bash

export CXX=clang
export CXXFLAGS="-Weverything -Werror \
    -Wno-c++98-compat \
    -Wno-exit-time-destructors \
    "
