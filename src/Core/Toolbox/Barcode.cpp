#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/Toolbox/Barcode.h>

#include <optional>

namespace py = pybind11;

namespace ZividPython
{
    using namespace Zivid::Experimental::Toolbox;

    template<typename Format>
    std::optional<BarcodeFormatFilter<Format>> formatSetToFilter(const std::set<Format> &formats)
    {
        std::optional<BarcodeFormatFilter<Format>> filter;
        for(const auto &format : formats)
        {
            filter = filter.has_value() ? (*filter | format) : BarcodeFormatFilter<Format>(format);
        }
        return filter;
    }

    void wrapEnum(pybind11::enum_<LinearBarcodeFormat> pyEnum)
    {
        pyEnum.value("code128", LinearBarcodeFormat::code128)
            .value("code93", LinearBarcodeFormat::code93)
            .value("code39", LinearBarcodeFormat::code39)
            .value("ean13", LinearBarcodeFormat::ean13)
            .value("ean8", LinearBarcodeFormat::ean8)
            .value("upcA", LinearBarcodeFormat::upcA)
            .value("upcE", LinearBarcodeFormat::upcE);
    }

    void wrapEnum(pybind11::enum_<MatrixBarcodeFormat> pyEnum)
    {
        pyEnum.value("qrcode", MatrixBarcodeFormat::qrcode).value("dataMatrix", MatrixBarcodeFormat::dataMatrix);
    }

    void wrapClass(pybind11::class_<ReleasableBarcodeDetector> pyClass)
    {
        pyClass.def(py::init())
            .def(
                "suggest_settings",
                [](ReleasableBarcodeDetector &detector, ReleasableCamera &camera) {
                    return detector.suggestSettings(camera.impl());
                },
                py::arg("camera"))
            .def(
                "read_linear_codes",
                [](ReleasableBarcodeDetector &detector,
                   const ReleasableFrame2D &frame2d,
                   const std::set<LinearBarcodeFormat> &formats) {
                    const auto filter = formatSetToFilter(formats);
                    if(filter.has_value())
                    {
                        return detector.readLinearCodes(frame2d.impl(), filter.value());
                    }
                    return detector.readLinearCodes(frame2d.impl(), LinearBarcodeFormatFilter::all());
                },
                py::arg("frame2d"),
                py::arg("formats"))
            .def(
                "read_matrix_codes",
                [](ReleasableBarcodeDetector &detector,
                   const ReleasableFrame2D &frame2d,
                   const std::set<MatrixBarcodeFormat> &formats) {
                    const auto filter = formatSetToFilter(formats);
                    if(filter.has_value())
                    {
                        return detector.readMatrixCodes(frame2d.impl(), filter.value());
                    }
                    return detector.readMatrixCodes(frame2d.impl(), MatrixBarcodeFormatFilter::all());
                },
                py::arg("frame2d"),
                py::arg("formats"));
    }

    void wrapClass(pybind11::class_<LinearBarcodeDetectionResult> pyClass)
    {
        pyClass.def("code", &LinearBarcodeDetectionResult::code)
            .def("code_format", [](LinearBarcodeDetectionResult &result) { return toString(result.codeFormat()); })
            .def("center_position", [](LinearBarcodeDetectionResult &result) {
                return std::array<float, 2>{ result.centerPosition().x, result.centerPosition().y };
            });
    }

    void wrapClass(pybind11::class_<MatrixBarcodeDetectionResult> pyClass)
    {
        pyClass.def("code", &MatrixBarcodeDetectionResult::code)
            .def("code_format", [](MatrixBarcodeDetectionResult &result) { return toString(result.codeFormat()); })
            .def("center_position", [](MatrixBarcodeDetectionResult &result) {
                return std::array<float, 2>{ result.centerPosition().x, result.centerPosition().y };
            });
    }
} // namespace ZividPython
