#pragma once

#include <Zivid/UnorganizedPointCloud.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableUnorganizedPointCloud : public Releasable<Zivid::UnorganizedPointCloud>
    {
    public:
        using Releasable<Zivid::UnorganizedPointCloud>::Releasable;

        ZIVID_PYTHON_FORWARD_0_ARGS(size)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            extended,
            const Zivid::UnorganizedPointCloud &,
            other)
        ZIVID_PYTHON_FORWARD_1_ARGS(extend, const Zivid::UnorganizedPointCloud &, other)
        ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            voxelDownsampled,
            float,
            voxelSize,
            int,
            minPointsPerVoxel)
        ZIVID_PYTHON_FORWARD_1_ARGS(transform, const Zivid::Matrix4x4 &, matrix)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableUnorganizedPointCloud,
            transformed,
            const Zivid::Matrix4x4 &,
            matrix)
        ZIVID_PYTHON_FORWARD_0_ARGS(center)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableUnorganizedPointCloud, clone, const);
    };

    void wrapClass(pybind11::class_<ReleasableUnorganizedPointCloud> pyClass);
} // namespace ZividPython
