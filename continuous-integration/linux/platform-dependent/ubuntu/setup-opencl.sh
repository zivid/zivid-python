#!/bin/bash

function ubuntu_install_opencl_cpu_runtime {

    # Download the key to system keyring
    INTEL_KEY_URL=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    wget -O- $INTEL_KEY_URL | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null || exit $?

    # Add signed entry to apt sources and configure the APT client to use Intel repository
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list || exit $?
    apt update || exit $?

    # Install the OpenCL runtime
    # TODO: remove libxml2 once Intel sorts out its package dependencies
    apt --assume-yes install libxml2 intel-oneapi-runtime-opencl-2024 intel-oneapi-runtime-compilers-2024 || exit $?

}

export DEBIAN_FRONTEND=noninteractive
ubuntu_install_opencl_cpu_runtime || exit $?
