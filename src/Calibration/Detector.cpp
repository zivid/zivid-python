#include <Zivid/Calibration/Detector.h>

#include <ZividPython/Calibration/Detector.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ZividPython
{

    void wrapEnum(pybind11::enum_<Zivid::Calibration::CalibrationBoardDetectionStatus> pyEnum)
    {
        pyEnum.value("ok", Zivid::Calibration::CalibrationBoardDetectionStatus::ok)
            .value(
                "no_valid_fiducial_marker_detected",
                Zivid::Calibration::CalibrationBoardDetectionStatus::noValidFiducialMarkerDetected)
            .value(
                "multiple_valid_fiducial_markers_detected",
                Zivid::Calibration::CalibrationBoardDetectionStatus::multipleValidFiducialMarkersDetected)
            .value("board_detection_failed", Zivid::Calibration::CalibrationBoardDetectionStatus::boardDetectionFailed)
            .value(
                "insufficient_3d_quality", Zivid::Calibration::CalibrationBoardDetectionStatus::insufficient3DQuality);
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResult> pyClass)
    {
        pyClass.def("valid", &Zivid::Calibration::DetectionResult::valid)
            .def(
                "centroid",
                [](const Zivid::Calibration::DetectionResult &detectionResult) {
                    return Conversion::toPyVector(detectionResult.centroid());
                })
            .def("pose", &Zivid::Calibration::DetectionResult::pose)
            .def("status", &Zivid::Calibration::DetectionResult::status)
            .def("status_description", &Zivid::Calibration::DetectionResult::statusDescription)
            .def(
                "feature_points",
                [](const Zivid::Calibration::DetectionResult &detectionResult) {
                    const auto nativePoints = detectionResult.featurePoints();
                    auto points = std::vector<Eigen::Vector3f>{};
                    points.reserve(nativePoints.size());
                    for(const auto &point : nativePoints)
                    {
                        points.emplace_back(Conversion::toPyVector(point));
                    }
                    return points;
                })
            .def("feature_points_2d", [](const Zivid::Calibration::DetectionResult &detectionResult) {
                const auto nativePoints = detectionResult.featurePoints2D();
                auto points = std::vector<Eigen::Vector2f>{};
                points.reserve(nativePoints.size());
                for(const auto &point : nativePoints)
                {
                    points.emplace_back(Conversion::toPyVector(point));
                }
                return points;
            });
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerShape> pyClass)
    {
        pyClass.def("id_", &Zivid::Calibration::MarkerShape::id)
            .def("pose", &Zivid::Calibration::MarkerShape::pose)
            .def(
                "corners_in_pixel_coordinates",
                [](const Zivid::Calibration::MarkerShape &markerShape) {
                    const auto nativeCorners = markerShape.cornersInPixelCoordinates();
                    auto corners = std::array<Eigen::Vector2f, 4>{};
                    for(int i = 0; i < 4; i++)
                    {
                        corners[i] = Conversion::toPyVector(nativeCorners[i]);
                    }
                    return corners;
                })
            .def("corners_in_camera_coordinates", [](const Zivid::Calibration::MarkerShape &markerShape) {
                const auto nativeCorners = markerShape.cornersInCameraCoordinates();
                auto corners = std::array<Eigen::Vector3f, 4>{};
                for(int i = 0; i < 4; i++)
                {
                    corners[i] = Conversion::toPyVector(nativeCorners[i]);
                }
                return corners;
            });
    }

#define ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, name)                                                        \
    pyClass.def_property_readonly_static(#name, [](py::object) { return Zivid::Calibration::MarkerDictionary::name; });

    void wrapClass(pybind11::class_<Zivid::Calibration::MarkerDictionary> pyClass)
    {
        pyClass.def("marker_count", &Zivid::Calibration::MarkerDictionary::markerCount);

        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco4x4_50);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco4x4_100);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco4x4_250);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco4x4_1000);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco5x5_50);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco5x5_100);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco5x5_250);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco5x5_1000);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco6x6_50);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco6x6_100);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco6x6_250);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco6x6_1000);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco7x7_50);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco7x7_100);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco7x7_250);
        ZIVID_PYTHON_WRAP_MARKER_DICTIONARY_NAME(pyClass, aruco7x7_1000);
    }

    void wrapClass(pybind11::class_<Zivid::Calibration::DetectionResultFiducialMarkers> pyClass)
    {
        pyClass.def("valid", &Zivid::Calibration::DetectionResultFiducialMarkers::valid)
            .def("allowed_marker_ids", &Zivid::Calibration::DetectionResultFiducialMarkers::allowedMarkerIds)
            .def("detected_markers", &Zivid::Calibration::DetectionResultFiducialMarkers::detectedMarkers);
    }
} // namespace ZividPython
