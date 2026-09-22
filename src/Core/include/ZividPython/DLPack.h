#pragma once

#include <Zivid/ComputeWrappers.h>
#include <Zivid/DeviceArray.h>

#include <dlpack/dlpack.h>

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

namespace ZividPython
{
    namespace DLPack
    {
        constexpr auto legacyCapsuleName = "dltensor";
        constexpr auto versionedCapsuleName = "dltensor_versioned";

        int cudaDeviceOrdinalForPointer(const void *devicePointer);

        void requireCUDABackend(Zivid::ComputeBackend backend);

        void requireExchangeableAsIs(const pybind11::kwargs &kwargs, int deviceOrdinal);

        bool consumerAcceptsVersionedTensor(const pybind11::kwargs &kwargs);

        void releaseLegacyCapsule(PyObject *capsule);

        void releaseVersionedCapsule(PyObject *capsule);

        template<typename Format>
        constexpr DLDataType dataTypeFor()
        {
            using ValueType = typename Format::ValueType;
            constexpr auto bits = static_cast<uint8_t>(sizeof(ValueType) * 8);
            constexpr uint16_t lanes = 1;
            if constexpr(std::is_same_v<ValueType, float>)
            {
                return DLDataType{ kDLFloat, bits, lanes };
            }
            else if constexpr(std::is_unsigned_v<ValueType>)
            {
                return DLDataType{ kDLUInt, bits, lanes };
            }
            else
            {
                static_assert(
                    std::is_integral_v<ValueType> && std::is_signed_v<ValueType>,
                    "Unsupported DeviceArray ValueType for DLPack conversion");
                return DLDataType{ kDLInt, bits, lanes };
            }
        }

        template<typename Format>
        struct ManagerContext
        {
            Zivid::DeviceArray<Format> deviceArray;
            std::vector<int64_t> shape;
            std::vector<int64_t> strides;
        };

        template<typename Format>
        std::unique_ptr<ManagerContext<Format>> createManagerContext(const Zivid::DeviceArray<Format> &deviceArray)
        {
            const auto shape = deviceArray.shape();
            const auto strides = deviceArray.strides();
            return std::make_unique<ManagerContext<Format>>(ManagerContext<Format>{
                deviceArray,
                std::vector<int64_t>{ shape.begin(), shape.end() },
                std::vector<int64_t>{ strides.begin(), strides.end() },
            });
        }

        template<typename Format>
        DLTensor describeTensor(ManagerContext<Format> &context, int deviceOrdinal)
        {
            DLTensor tensor{};
            tensor.data = context.deviceArray.devicePointer();
            tensor.device = DLDevice{ kDLCUDA, deviceOrdinal };
            tensor.ndim = static_cast<int32_t>(context.shape.size());
            tensor.dtype = dataTypeFor<Format>();
            tensor.shape = context.shape.data();
            tensor.strides = context.strides.data();
            tensor.byte_offset = 0;
            return tensor;
        }

        // This is the deleter the consumer calls, so it must match the release() in the create
        // functions.
        template<typename Format, typename Tensor>
        void deleteManagedTensor(Tensor *managedTensor)
        {
            const std::unique_ptr<ManagerContext<Format>> context{ static_cast<ManagerContext<Format> *>(
                managedTensor->manager_ctx) };
            const std::unique_ptr<Tensor> owned{ managedTensor };
        }

        template<typename Format>
        DLManagedTensor *createLegacyTensor(const Zivid::DeviceArray<Format> &deviceArray, int deviceOrdinal)
        {
            auto context = createManagerContext(deviceArray);
            auto managedTensor = std::make_unique<DLManagedTensor>();
            managedTensor->dl_tensor = describeTensor(*context, deviceOrdinal);
            managedTensor->manager_ctx = context.release();
            managedTensor->deleter = &deleteManagedTensor<Format, DLManagedTensor>;
            return managedTensor.release();
        }

        template<typename Format>
        DLManagedTensorVersioned *createVersionedTensor(
            const Zivid::DeviceArray<Format> &deviceArray,
            int deviceOrdinal)
        {
            auto context = createManagerContext(deviceArray);
            auto managedTensor = std::make_unique<DLManagedTensorVersioned>();
            managedTensor->version = DLPackVersion{ DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION };
            managedTensor->dl_tensor = describeTensor(*context, deviceOrdinal);
            managedTensor->flags = 0;
            managedTensor->manager_ctx = context.release();
            managedTensor->deleter = &deleteManagedTensor<Format, DLManagedTensorVersioned>;
            return managedTensor.release();
        }
    } // namespace DLPack

    template<typename ReleasableType>
    void addDLPackMethods(pybind11::class_<ReleasableType> &pyClass)
    {
        pyClass
            .def(
                "__dlpack__",
                [](const ReleasableType &self, const pybind11::kwargs &kwargs) {
                    DLPack::requireCUDABackend(self.backend());
                    const auto deviceOrdinal = DLPack::cudaDeviceOrdinalForPointer(self.devicePointer());
                    DLPack::requireExchangeableAsIs(kwargs, deviceOrdinal);
                    if(DLPack::consumerAcceptsVersionedTensor(kwargs))
                    {
                        return pybind11::capsule(
                            DLPack::createVersionedTensor(self.impl(), deviceOrdinal),
                            DLPack::versionedCapsuleName,
                            &DLPack::releaseVersionedCapsule);
                    }
                    return pybind11::capsule(
                        DLPack::createLegacyTensor(self.impl(), deviceOrdinal),
                        DLPack::legacyCapsuleName,
                        &DLPack::releaseLegacyCapsule);
                })
            .def("__dlpack_device__", [](const ReleasableType &self) {
                DLPack::requireCUDABackend(self.backend());
                return pybind11::make_tuple(
                    static_cast<int32_t>(kDLCUDA), DLPack::cudaDeviceOrdinalForPointer(self.devicePointer()));
            });
    }
} // namespace ZividPython
