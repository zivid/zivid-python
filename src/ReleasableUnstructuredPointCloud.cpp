#include <ZividPython/ReleasableUnstructuredPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;


namespace
{
    template<typename NativeType, typename WrapperType, int nChannels>
    py::array_t<WrapperType> getUnstructuredDataAsNumpyArray(const ZividPython::ReleasableUnstructuredPointCloud &pc)
    {
        static_assert(nChannels > 0, "Channel count must be one or higher");
        static_assert(sizeof(WrapperType) * nChannels == sizeof(NativeType), "Unexpected data format");

        const int nPoints = pc.impl().size();

        // Allocate raw memory that we can later give the numpy array ownership of
        WrapperType* data = new WrapperType[nPoints*nChannels];

        // Copy data from GPU
        pc.impl().copyData<NativeType>(reinterpret_cast<NativeType*>(data));


        // Wrap data as numpy array and give it ownership
        return py::array_t<WrapperType>(
                {nPoints, nChannels},
                {sizeof(WrapperType) * nChannels, sizeof(WrapperType)},
                data,
                py::capsule(data, [](void *ptr) {
                    delete[] static_cast<WrapperType*>(ptr);
                })
            );

    }
}

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableUnstructuredPointCloud> pyClass)
    {
        pyClass.def("size", &ReleasableUnstructuredPointCloud::size)
            .def("copy_points_xyz", [](const ReleasableUnstructuredPointCloud &pc){return getUnstructuredDataAsNumpyArray<Zivid::PointXYZ, float, 3>(pc);})
            .def("copy_colors_rgba", [](const ReleasableUnstructuredPointCloud &pc){return getUnstructuredDataAsNumpyArray<Zivid::ColorRGBA, uint8_t, 4>(pc);})
            .def("copy_snrs", [](const ReleasableUnstructuredPointCloud &pc){return getUnstructuredDataAsNumpyArray<Zivid::SNR, float, 1>(pc);});
    }
} // namespace ZividPython
