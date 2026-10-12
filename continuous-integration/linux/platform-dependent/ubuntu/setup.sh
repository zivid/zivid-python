#!/bin/bash

export DEBIAN_FRONTEND=noninteractive

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
    gpg \
    || exit $?

update-alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?
