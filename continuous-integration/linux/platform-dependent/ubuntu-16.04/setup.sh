#!/bin/bash

if [ -z "$1" ]
then
    echo "Usage: $0 zivid-sdk-version"
    exit 1
fi
zividSDKVersion=$1

export DEBIAN_FRONTEND=noninteractive

function apt-yes {
    apt-get --assume-yes "$@"
}

apt-yes update || exit $?
apt-yes dist-upgrade || exit $?

apt-yes install software-properties-common || exit $?
add-apt-repository -y ppa:ubuntu-toolchain-r/test || exit $?
apt-yes update || exit $?

apt-yes install \
    alien \
    g++-9 \
    python3-dev \
    python3-venv \
    python3-pip \
    wget \
    || exit $?

update-alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 0 || exit $?
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 0 || exit $?

function install_opencl_cpu_runtime {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-setup-opencl-cpu-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget --progress=bar:force:noscroll https://www.dropbox.com/s/0cvg8fypylgal2m/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz || exit $?
    tar -xf opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz || exit $?
    alien -i opencl_runtime_*/rpm/*.rpm || exit $?
    mkdir -p /etc/OpenCL/vendors || exit $?
    ls /opt/intel/opencl*/lib64/libintelocl.so > /etc/OpenCL/vendors/intel.icd || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_opencl_cpu_runtime || exit $?

function install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    chmod o+x "$TMP_DIR" || exit $?
    pushd $TMP_DIR || exit $?
    wget --progress=bar:force:noscroll "$@" || exit $?
    apt-yes install --fix-broken ./*deb || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/$zividSDKVersion/u16/zivid-telicam-driver_2.0.0.1-1_amd64.deb || exit $?
install_www_deb https://www.zivid.com/hubfs/softwarefiles/releases/$zividSDKVersion/u16/zivid_${zividSDKVersion}_amd64.deb || exit $?
