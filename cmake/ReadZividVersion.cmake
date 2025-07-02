# Reads the zivid sdk version from a json file. Sets `ZIVID_SDK_VERSION` CMake variable in parent
# scope to the version read from the file.
function(zivid_read_sdk_version)
    cmake_parse_arguments("ARG" "" "JSON_FILE" "" ${ARGN})
    file(READ "${ARG_JSON_FILE}" SDK_VERSION_JSON)
    message(STATUS "Reading Zivid SDK version from ${ARG_JSON_FILE}")
    message(STATUS "SDK version JSON content: ${SDK_VERSION_JSON}")
    string(JSON MAJOR_VERSION GET "${SDK_VERSION_JSON}" major)
    string(JSON MINOR_VERSION GET "${SDK_VERSION_JSON}" minor)
    string(JSON PATCH_VERSION GET "${SDK_VERSION_JSON}" patch)
    set(ZIVID_SDK_VERSION "${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}" PARENT_SCOPE)
endfunction()
