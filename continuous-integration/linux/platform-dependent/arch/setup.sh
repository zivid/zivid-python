#!/bin/bash

pacman -Syu --noconfirm --needed \
       clang \
       clinfo \
       cmake \
       diffutils \
       fakeroot \
       git \
       intel-tbb \
       ncurses \
       ninja \
       numactl \
       python-pip \
       shellcheck \
       sudo \
    || exit $?

function aur_install {
    PACKAGE=$1; shift
    VERSION_HASH=$1; shift
    IGNORE_DEPS=$*
    TMP_DIR=$(sudo -u nobody mktemp --tmpdir --directory zivid-python-aur-install-XXXX) || exit $?
    git clone https://aur.archlinux.org/$PACKAGE.git $TMP_DIR || exit $?
    if [[ -n $VERSION_HASH ]] ; then
        git --git-dir="$TMP_DIR/.git" --work-tree="$TMP_DIR" checkout $VERSION_HASH || exit $?
    fi
    pushd $TMP_DIR || exit $?
    for dep in $IGNORE_DEPS; do
        sed -i s/\'$dep\'//g PKGBUILD || exit $?
    done || exit $?
    PKGEXT=.pkg.tar sudo -E -u nobody makepkg || exit $?
    pacman -U --noconfirm ./*$PACKAGE*.tar || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

# Use so file from ncurses instead of ncurses5-compat-libs
# as dependency for intel-opencl-runtime
ln -s /usr/lib/libtinfo.so.{6,5} || exit $?
aur_install intel-opencl-runtime a7db4fe8cfa872078034f7966bb2def788bf8e5d ncurses5-compat-libs || exit $?

aur_install zivid-telicam-driver c2dc70bc1118c6aef729bb1804ced8e2c74cbc5a || exit $?
aur_install zivid 9d5f15f1cf9bb43445378abc4d443c1c8b79451d || exit $?
