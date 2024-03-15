
#include <ZividPython/Projection.h>

#include <Zivid/Projection/Projection.h>
#include <Zivid/Resolution.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableProjectedImage.h>

#include <array>
#include <vector>

namespace ZividPython::Projection
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        dest.def("projector_resolution", [](const ReleasableCamera &camera) {
            const auto resolution = Zivid::Projection::projectorResolution(camera.impl());
            return std::make_pair(resolution.height(), resolution.width());
        });

        dest.def(
            "show_image_bgra",
            [](ReleasableCamera &camera,
               const pybind11::array_t<uint8_t, pybind11::array::c_style | pybind11::array::forcecast> imageBGRA) {
                const auto info = imageBGRA.request();

                if(info.ndim != 3)
                {
                    throw std::runtime_error("Input image array must be three dimensional.");
                }

                const auto height = info.shape[0];
                const auto width = info.shape[1];
                const auto channels = info.shape[2];

                if(channels != 4)
                {
                    throw std::runtime_error("Input image array must have four color channels (BGRA).");
                }

                const auto resolution = Zivid::Resolution{ static_cast<size_t>(width), static_cast<size_t>(height) };
                const Zivid::Image<Zivid::ColorBGRA> zividImage{
                    resolution, imageBGRA.data(), imageBGRA.data() + resolution.size() * sizeof(Zivid::ColorBGRA)
                };
                auto projectedImage = Zivid::Projection::showImage(camera.impl(), zividImage);
                return ZividPython::ReleasableProjectedImage(std::move(projectedImage));
            });

        dest.def("pixels_from_3d_points",
                 [](const ReleasableCamera &camera, const std::vector<std::array<float, 3>> points) {
                     auto pointsInternal = std::vector<Zivid::PointXYZ>();
                     pointsInternal.reserve(points.size());
                     std::transform(points.begin(),
                                    points.end(),
                                    std::back_inserter(pointsInternal),
                                    [](const auto &point) {
                                        return Zivid::PointXYZ{ point[0], point[1], point[2] };
                                    });

                     const auto outputInternal = Zivid::Projection::pixelsFrom3DPoints(camera.impl(), pointsInternal);

                     auto output = std::vector<std::array<float, 2>>();
                     output.reserve(outputInternal.size());
                     std::transform(outputInternal.begin(),
                                    outputInternal.end(),
                                    std::back_inserter(output),
                                    [](const auto &pointxy) {
                                        return std::array<float, 2>{ pointxy.x, pointxy.y };
                                    });
                     return output;
                 });
    }
} // namespace ZividPython::Projection
