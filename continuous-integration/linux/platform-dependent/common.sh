#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

function install_opencl_cpu_runtime {
    TMP_DIR=$(mktemp --tmpdir --directory zivid-setup-opencl-cpu-XXXX) || exit $?
    pushd $TMP_DIR || exit $?
    wget -q https://www.dropbox.com/s/h0txd04aqfluglq/l_opencl_p_18.1.0.015.tgz || exit $?
    tar -xf l_opencl_p_18.1.0.015.tgz || exit $?
    cd l_opencl_*/ || exit $?

    cat > installer_config.cfg <<EOF
# See silent.cfg in the .tgz for description of the options.
ACCEPT_EULA=accept
# 'yes' below is required because the installer officially supports only Ubuntu 16.04. However, it will
# work fine on Ubuntu 18.04 and Fedora 30 as well.
CONTINUE_WITH_OPTIONAL_ERROR=yes
PSET_INSTALL_DIR=/opt/intel
CONTINUE_WITH_INSTALLDIR_OVERWRITE=yes
COMPONENTS=DEFAULTS
PSET_MODE=install
INTEL_SW_IMPROVEMENT_PROGRAM_CONSENT=no
SIGNING_ENABLED=yes
EOF
    echo "Running Intel OpenCL driver installer."
    echo "Note: Installer will warn about 'Unsupported operating system' if not run on Ubuntu 16.04."
    echo "This warning can be ignored."
    echo
    ./install.sh --silent installer_config.cfg || exit $?
    popd || exit $?
    rm -r $TMP_DIR || exit $?
}

# Read versions.json and set as environment variables
VERSIONS_FILE="${SCRIPT_DIR}/../../versions.json"
for var in $(jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]" ${VERSIONS_FILE} ); do
    echo "Setting env var from ${VERSIONS_FILE}: ${var}"
    export ${var?}
done
