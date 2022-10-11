#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

dnf --assumeyes install \
    bsdtar \
    clinfo \
    g++ \
    libatomic \
    python3-devel \
    python3-pip \
    wget \
    || exit $?

alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?

source $(realpath $SCRIPT_DIR/../common.sh) || exit $?
# Install OpenCL CPU runtime driver prerequisites
dnf --assumeyes install \
    numactl-libs \
    redhat-lsb-core \
    || exit $?
install_opencl_cpu_runtime || exit $?

function install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    ar x ./*deb || exit $?
    bsdtar -xf data.tar.*z -C / || exit $?
    ldconfig || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/2.8.0+891708ba-1/u18/zivid-telicam-driver_3.0.1.1-3_amd64.deb || exit $?
install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/2.8.0+891708ba-1/u18/zivid_2.8.0+891708ba-1_amd64.deb || exit $?