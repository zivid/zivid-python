#!/bin/bash

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

fedora_install_opencl_cpu_runtime || exit $?
