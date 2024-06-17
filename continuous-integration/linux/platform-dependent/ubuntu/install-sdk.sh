#!/bin/bash

function ubuntu_install_www_deb {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-python-install-www-deb-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -nv "$@" || exit $?
    apt-get --assume-yes install --fix-broken ./*deb || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

ubuntu_install_www_deb "https://downloads.zivid.com/sdk/previews/${ZIVID_SDK_EXACT_VERSION}/u20/amd64/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?
