#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace
{
    template<typename NativeType, typename WrapperType, int nChannels>
    py::array_t<WrapperType> getUnorganizedDataAsNumpyArray(const ZividPython::ReleasableUnorganizedPointCloud &pc)
    {
        static_assert(nChannels > 0, "Channel count must be one or higher");
        static_assert(sizeof(WrapperType) * nChannels == sizeof(NativeType), "Unexpected data format");

        const int nPoints = pc.impl().size();

        // Allocate raw memory that we can later give the numpy array ownership of
        WrapperType *data = new WrapperType[nPoints * nChannels];

        // Copy data from GPU
        pc.impl().copyData<NativeType>(reinterpret_cast<NativeType *>(data));

        // Wrap data as numpy array and give it ownership
        return py::array_t<WrapperType>({ nPoints, nChannels },
                                        { sizeof(WrapperType) * nChannels, sizeof(WrapperType) },
                                        data,
                                        py::capsule(data, [](void *ptr) { delete[] static_cast<WrapperType *>(ptr); }));
    }
} // namespace

namespace ZividPython
{
    void wrapClass(pybind11::class_<ReleasableUnorganizedPointCloud> pyClass)
    {
        pyClass.def("size", &ReleasableUnorganizedPointCloud::size)
            .def("copy_points_xyz",
                 [](const ReleasableUnorganizedPointCloud &pc) {
                     return getUnorganizedDataAsNumpyArray<Zivid::PointXYZ, float, 3>(pc);
                 })
            .def("copy_colors_rgba",
                 [](const ReleasableUnorganizedPointCloud &pc) {
                     return getUnorganizedDataAsNumpyArray<Zivid::ColorRGBA, uint8_t, 4>(pc);
                 })
            .def("copy_colors_bgra",
                 [](const ReleasableUnorganizedPointCloud &pc) {
                     return getUnorganizedDataAsNumpyArray<Zivid::ColorBGRA, uint8_t, 4>(pc);
                 })
            .def("copy_snrs",
                 [](const ReleasableUnorganizedPointCloud &pc) {
                     return getUnorganizedDataAsNumpyArray<Zivid::SNR, float, 1>(pc);
                 })
            .def(
                "extended",
                [](ReleasableUnorganizedPointCloud &pc, const ReleasableUnorganizedPointCloud &other) {
                    return pc.extended(other.impl());
                },
                py::arg("other"))
            .def("voxel_downsampled",
                 &ReleasableUnorganizedPointCloud::voxelDownsampled,
                 py::arg("voxel_size"),
                 py::arg("min_points_per_voxel"))
            .def("transform",
                 [](ReleasableUnorganizedPointCloud &pointCloud,
                    const Eigen::Matrix<float, 4, 4, Eigen::RowMajor> &matrix) {
                     pointCloud.transform(Conversion::toCpp(matrix));
                 });
    }
} // namespace ZividPython
