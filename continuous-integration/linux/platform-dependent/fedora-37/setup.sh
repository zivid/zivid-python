#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

dnf --assumeyes install \
    bsdtar \
    clinfo \
    g++ \
    jq \
    libatomic \
    python3-devel \
    python3-pip \
    wget \
    || exit $?

alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?

source $(realpath $SCRIPT_DIR/../common.sh) || exit $?
fedora_install_opencl_cpu_runtime || exit $?

function install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    ar x ./*deb || exit $?
    bsdtar -xf data.tar.* -C / || exit $?
    ldconfig || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/u20/zivid-telicam-driver_${ZIVID_TELICAM_EXACT_VERSION}_amd64.deb" || exit $?
install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/u20/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?