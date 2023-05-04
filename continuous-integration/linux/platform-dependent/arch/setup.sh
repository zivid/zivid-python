#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
       python-setuptools \
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

aur_install zivid-telicam-driver 4fd3efa4b01a0ff2142928f38ed50c1a9fbfcb44 || exit $?
aur_install zivid 54bca017fe77f2e415761881a602d39097745c15 || exit $?

# Install custom conan settings.yml
# TODO(https://github.com/conan-io/conan/pull/13007)
echo "Updating conan settings"
mkdir --parents /root/.conan/ || exit $?
cp $SCRIPT_DIR/settings.yml /root/.conan/ || exit $?

# Use cloudflare mirror
# The dollar sign in variables should not be expanded, so disable this check to use single quotes
# shellcheck disable=SC2016
echo 'Server = https://cloudflaremirrors.com/archlinux/$repo/os/$arch' >/etc/pacman.d/mirrorlist