#!/bin/bash

source /etc/os-release || exit $?

# Zivid SDK 2.11 is published for Ubuntu 20.04 (u20) and 22.04 (u22) only
if [[ "$VERSION_ID" == "20.04" ]]; then
    ZIVID_UBUNTU_DIR=u20
else
    ZIVID_UBUNTU_DIR=u22
fi

function ubuntu_install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    apt-get --assume-yes install --fix-broken ./*deb || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

ubuntu_install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/${ZIVID_UBUNTU_DIR}/zivid-telicam-driver_${ZIVID_TELICAM_EXACT_VERSION}_amd64.deb" || exit $?
ubuntu_install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/${ZIVID_UBUNTU_DIR}/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?
