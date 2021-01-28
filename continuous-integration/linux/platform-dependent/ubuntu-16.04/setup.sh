#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

function apt-yes {
    apt-get --assume-yes "$@"
}

apt-yes update || exit $?
apt-yes dist-upgrade || exit $?

apt-yes install software-properties-common || exit $?
add-apt-repository -y ppa:ubuntu-toolchain-r/test || exit $?
add-apt-repository --yes ppa:deadsnakes/ppa
apt-yes update || exit $?

apt-yes install \
    clinfo \
    g++-9 \
    python3.6-dev \
    python3.6-venv \
    wget \
    curl \
    || exit $?

curl https://bootstrap.pypa.io/get-pip.py | python3.6

source "$SCRIPT_DIR/../common.sh" || exit $?
# Install OpenCL CPU runtime driver prerequisites
apt-yes install libnuma-dev lsb-core || exit $?
install_opencl_cpu_runtime || exit $?

update-alternatives --install /usr/bin/python python /usr/bin/python3.6 0 || exit $?
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.6 0 || exit $?
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 0 || exit $?
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 0 || exit $?

function install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    apt-yes install --fix-broken ./*deb || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/2.2.0+f0867d62-1/u16/zivid-telicam-driver_3.0.1.1-3_amd64.deb || exit $?
install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/2.2.0+f0867d62-1/u16/zivid_2.2.0+f0867d62-1_amd64.deb || exit $?
