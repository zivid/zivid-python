#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

function apt-yes {
    apt-get --assume-yes "$@"
}

apt-yes update || exit $?
apt-yes dist-upgrade || exit $?

apt-yes install \
    clinfo \
    g++ \
    jq \
    python3-dev \
    python3-venv \
    python3-pip \
    wget \
    || exit $?

update-alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?

source $(realpath $SCRIPT_DIR/../common.sh) || exit $?
# Install OpenCL CPU runtime driver prerequisites
apt-yes install libnuma-dev lsb-core || exit $?
install_opencl_cpu_runtime || exit $?

function install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    apt-yes install --fix-broken ./*deb || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_www_deb "https://www.zivid.com/hubfs/softwarefiles/releases/${ZIVID_SDK_EXACT_VERSION}/u22/zivid-telicam-driver_${ZIVID_TELICAM_EXACT_VERSION}_amd64.deb" || exit $?
install_www_deb "https://www.zivid.com/hubfs/softwarefiles/releases/${ZIVID_SDK_EXACT_VERSION}/u22/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?
