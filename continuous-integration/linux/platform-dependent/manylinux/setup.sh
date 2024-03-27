#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

dnf --assumeyes install \
    bsdtar \
    jq \
    libatomic \
    wget \
    || exit $?


source $(realpath $SCRIPT_DIR/../common.sh) || exit $?

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

install_www_deb "https://downloads.zivid.com/sdk/previews/${ZIVID_SDK_EXACT_VERSION}/u20/amd64/zivid_${ZIVID_SDK_EXACT_VERSION}_amd64.deb" || exit $?
