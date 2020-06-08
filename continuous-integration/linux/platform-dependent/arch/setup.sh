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
        sed -i s/\'$dep\'//g PKGBUILD
    done
    PKGEXT=.pkg.tar sudo -E -u nobody makepkg || exit $?
    pacman -U --noconfirm ./*$PACKAGE*.tar || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

# Use so file from ncurses instead of ncurses5-compat-libs
# as dependency for intel-opencl-runtime
ln -s /usr/lib/libtinfo.so.{6,5} || exit $?
aur_install intel-opencl-runtime "" ncurses5-compat-libs || exit $?

aur_install zivid-telicam-driver 48da46ee95cb679fd25149c251872a64817ff9ad || exit $?
aur_install zivid f9c38eab6467f0d98f4602f0a5af0f90be07a37d || exit $?
