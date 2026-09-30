#include <ZividPython/DLPack.h>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{
    constexpr int cuPointerAttributeDeviceOrdinal = 9;

#ifdef _WIN32
#    define ZIVID_CUDA_DRIVER_API __stdcall
#else
#    define ZIVID_CUDA_DRIVER_API
#endif

    using CuPointerGetAttribute = int(ZIVID_CUDA_DRIVER_API *)(void *, int, unsigned long long);

    CuPointerGetAttribute resolveCuPointerGetAttribute()
    {
#ifdef _WIN32
        auto *driverModule = GetModuleHandleA("nvcuda.dll");
        if(driverModule == nullptr)
        {
            throw std::runtime_error{
                "nvcuda.dll is not loaded into this process, so the CUDA driver entry point needed for the DLPack "
                "device ordinal cannot be resolved."
            };
        }
        auto *symbol = GetProcAddress(driverModule, "cuPointerGetAttribute");
#else
        auto *symbol = dlsym(RTLD_DEFAULT, "cuPointerGetAttribute");
#endif
        if(symbol == nullptr)
        {
            throw std::runtime_error{
                "Could not resolve cuPointerGetAttribute from the CUDA driver loaded into this process, so the DLPack "
                "device ordinal cannot be determined."
            };
        }
        return reinterpret_cast<CuPointerGetAttribute>(symbol);
    }

    template<typename Tensor>
    void releaseCapsuleAs(PyObject *capsule, const char *name)
    {
        if(PyCapsule_IsValid(capsule, name) == 0)
        {
            return;
        }
        auto *managedTensor = static_cast<Tensor *>(PyCapsule_GetPointer(capsule, name));
        managedTensor->deleter(managedTensor);
    }
} // namespace

namespace ZividPython::DLPack
{
    int cudaDeviceOrdinalForPointer(const void *devicePointer)
    {
        static const auto cuPointerGetAttribute = resolveCuPointerGetAttribute();

        int deviceOrdinal = -1;
        const auto result = cuPointerGetAttribute(
            &deviceOrdinal,
            cuPointerAttributeDeviceOrdinal,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(devicePointer)));
        if(result != 0)
        {
            throw std::runtime_error{ "cuPointerGetAttribute failed with CUresult " + std::to_string(result)
                                      + " while determining the DLPack device ordinal for a Zivid device pointer." };
        }
        return deviceOrdinal;
    }

    void requireCUDABackend(Zivid::ComputeBackend backend)
    {
        if(backend != Zivid::ComputeBackend::cuda)
        {
            throw std::runtime_error{
                "The DLPack protocol is only available on the CUDA backend. On the OpenCL backend device_pointer() "
                "returns a cl_mem handle, which DLPack cannot represent."
            };
        }
    }

    void requireExchangeableAsIs(const pybind11::kwargs &kwargs, int deviceOrdinal)
    {
        // Producer obligations from the array API specification, which are separate from version
        // negotiation:
        //   dl_device  - a device we cannot produce must raise BufferError. We only ever hand out
        //                the buffer where it already lives, so only our own device is accepted.
        //   copy=True  - the consumer wants memory it can mutate in isolation. We hand out a view
        //                onto memory other Zivid objects still read, and the wrapper has no CUDA
        //                copy primitive, so this must raise rather than quietly alias.
        //   copy=False - must never copy and must not raise. We never copy, so it is always honoured.
        if(kwargs.contains("dl_device") && !kwargs["dl_device"].is_none())
        {
            const auto requested = pybind11::tuple(kwargs["dl_device"]);
            const auto deviceType = pybind11::cast<int32_t>(requested[0]);
            const auto requestedOrdinal = pybind11::cast<int32_t>(requested[1]);
            if(deviceType != kDLCUDA || requestedOrdinal != deviceOrdinal)
            {
                throw pybind11::buffer_error{
                    "A Zivid DeviceArray can only be exchanged on the CUDA device it already lives on, ("
                    + std::to_string(static_cast<int32_t>(kDLCUDA)) + ", " + std::to_string(deviceOrdinal)
                    + "), but dl_device requested (" + std::to_string(deviceType) + ", "
                    + std::to_string(requestedOrdinal) + ")."
                };
            }
        }

        if(kwargs.contains("copy") && !kwargs["copy"].is_none() && pybind11::cast<bool>(kwargs["copy"]))
        {
            throw pybind11::buffer_error{
                "copy=True requires handing out an independent copy of the GPU buffer, which the Zivid "
                "Python wrapper cannot produce. Import the buffer without copy=True and copy it with the "
                "consuming framework, or use copy_to_host_organized_array/copy_to_host_unorganized_array."
            };
        }
    }

    bool consumerAcceptsVersionedTensor(const pybind11::kwargs &kwargs)
    {
        // Producer protocol from the array API specification:
        //   https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
        // No max_version means the consumer predates DLPack 1.0 and expects the legacy tensor. A
        // consumer on our major version or newer gets DLManagedTensorVersioned stamped with our own
        // version, including when its minor is lower than ours. A consumer on an older major gets
        // the legacy tensor, which we still implement, so version negotiation alone never has to
        // raise BufferError. The dl_device and copy obligations do, see requireExchangeableAsIs.
        //
        // Serving an older-major consumer the legacy tensor is only right while our major is 1,
        // where the only older major is 0. On a bump to 2.x a 1.x consumer would need a 1.x
        // versioned tensor we no longer produce, so that case must be handled deliberately
        // instead of silently falling back here.
        static_assert(DLPACK_MAJOR_VERSION == 1, "DLPack major version bumped: revisit __dlpack__ negotiation");

        if(!kwargs.contains("max_version"))
        {
            return false;
        }
        const auto maxVersion = kwargs["max_version"];
        if(maxVersion.is_none())
        {
            return false;
        }
        const auto major = pybind11::cast<uint32_t>(pybind11::tuple(maxVersion)[0]);
        return major >= DLPACK_MAJOR_VERSION;
    }

    void releaseLegacyCapsule(PyObject *capsule)
    {
        releaseCapsuleAs<DLManagedTensor>(capsule, legacyCapsuleName);
    }

    void releaseVersionedCapsule(PyObject *capsule)
    {
        releaseCapsuleAs<DLManagedTensorVersioned>(capsule, versionedCapsuleName);
    }
} // namespace ZividPython::DLPack
