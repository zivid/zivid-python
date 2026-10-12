#!/bin/bash

function fedora_install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    ar x ./*deb || exit $?
    bsdtar -xf data.tar.* -C / || exit $?
    ldconfig || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

fedora_install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/u20/zivid-telicam-driver_${ZIVID_TELICAM_EXACT_VERSION}_amd64.deb" || exit $?
fedora_install_www_deb "https://downloads.zivid.com/sdk/releases/${ZIVID_SDK_EXACT_VERSION}/u20/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?
