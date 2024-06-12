#include <Zivid/Calibration/Detector.h>

#include <ZividPython/Calibration/Detector.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResult> pyClass)
    {
        pyClass.def("valid", &Zivid::Calibration::DetectionResult::valid)
            .def("centroid",
                 [](const Zivid::Calibration::DetectionResult &detectionResult) {
                     return Conversion::toPyVector(detectionResult.centroid());
                 })
            .def("pose", &Zivid::Calibration::DetectionResult::pose);
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerShape> pyClass)
    {
        pyClass.def("id", &Zivid::Calibration::MarkerShape::id)
            .def("pose", &Zivid::Calibration::MarkerShape::pose)
            .def("corners_in_pixel_coordinates", [](const Zivid::Calibration::MarkerShape &markerShape){
                const auto nativeCorners = markerShape.cornersInPixelCoordinates();
                auto corners = std::array<Eigen::Vector2f, 4>{};
                for(int i=0; i<4; i++)
                {
                    corners[i] = Conversion::toPyVector(nativeCorners[i]);
                }
                return corners;
            })
            .def("corners_in_camera_coordinates", [](const Zivid::Calibration::MarkerShape &markerShape){
                const auto nativeCorners = markerShape.cornersInCameraCoordinates();
                auto corners = std::array<Eigen::Vector3f, 4>{};
                for(int i=0; i<4; i++)
                {
                    corners[i] = Conversion::toPyVector(nativeCorners[i]);
                }
                return corners;
            });
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResultFiducialMarkers> pyClass)
    {
        pyClass.def("valid", &Zivid::Calibration::DetectionResultFiducialMarkers::valid)
            .def("allowed_marker_ids", &Zivid::Calibration::DetectionResultFiducialMarkers::allowedMarkerIds)
            .def("detected_markers", &Zivid::Calibration::DetectionResultFiducialMarkers::detectedMarkers);
    }
} // namespace ZividPython
