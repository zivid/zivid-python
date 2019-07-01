# Compiler Installation

This document contains instruction for how to install a compatible compiler for compilation of the Zivid Python package.

## Ubuntu 18.04

    apt install g++

## Fedora 30

    dnf install g++

## Arch Linux

    pacman -S gcc

## Windows 10

Install Visual Studio Community Edition *2017* or similar edition. This can be done using the [Visual Studio Community 2019 installer](https://visualstudio.microsoft.com/vs/community/)

## Ubuntu 16.04

Ubuntu 16.04 does not contain an up to date compiler in the official repositories. The easiest way to install a newer compiler is to enable the [`ubuntu-toolchain-r/test` ppa](https://wiki.ubuntu.com/ToolChain#PPA_packages) and use it to install GCC 9.

    apt install software-properties-common
    add-apt-repository ppa:ubuntu-toolchain-r/test
    apt update
    apt install g++-9

To set GCC 9 as the default compiler execute the following two commands.

    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 0
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 0

An alternative to changing the default system compiler is to set the appropriate environment variables before building. This can be done the following way.

    export CXX=g++-9
    export CC=gcc
    # Run the install command
