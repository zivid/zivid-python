#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

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

function fedora_install_opencl_cpu_runtime {
    tee > /etc/yum.repos.d/oneAPI.repo << EOF
[oneAPI]
name=Intel® oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF
    dnf --assumeyes install intel-oneapi-runtime-opencl-2024 intel-oneapi-runtime-compilers-2024 || exit $?
}

# Read versions.json and set as environment variables
VERSIONS_FILE="${SCRIPT_DIR}/../../versions.json"
for var in $(jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]" ${VERSIONS_FILE} ); do
    echo "Setting env var from ${VERSIONS_FILE}: ${var}"
    export ${var?}
done
