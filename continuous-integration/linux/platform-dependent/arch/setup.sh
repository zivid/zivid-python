#!/bin/bash

pacman -Syu --noconfirm --needed \
       clang \
       cmake \
       diffutils \
       fakeroot \
       git \
       intel-tbb \
       ncurses \
       ninja \
       numactl \
       pybind11 \
       python-pip \
       shellcheck \
       sudo \
    || exit $?

function aur_install {
    PACKAGE=$1; shift
    IGNORE_DEPS=$*
    TMP_DIR=$(sudo -u nobody mktemp --tmpdir --directory zivid-python-aur-install-XXXX) || exit $?
    git clone https://aur.archlinux.org/$PACKAGE.git $TMP_DIR || exit $?
    pushd $TMP_DIR || exit $?
    for dep in $IGNORE_DEPS; do
        sed -i s/\'$dep\'//g PKGBUILD
    done
    PKGEXT=.tar sudo -E -u nobody makepkg || exit $?
    pacman -U --noconfirm ./*$PACKAGE*.tar || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

# Use so file from ncurses instead of ncurses5-compat-libs
# as dependency for intel-opencl-runtime
ln -s /usr/lib/libtinfo.so.{6,5} || exit $?
aur_install intel-opencl-runtime ncurses5-compat-libs || exit $?

aur_install zivid-telicam-sdk || exit $?
aur_install zivid || exit $?
