#pragma once

#include <Zivid/Experimental/Toolbox/Barcode.h>

#include <ZividPython/Releasable.h>

namespace ZividPython
{
    class ReleasableBarcodeDetector : public Releasable<Zivid::Experimental::Toolbox::BarcodeDetector>
    {
    public:
        using Releasable<Zivid::Experimental::Toolbox::BarcodeDetector>::Releasable;
        ZIVID_PYTHON_FORWARD_1_ARGS(suggestSettings, Zivid::Camera &, camera)
        ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(
            std::vector<Zivid::Experimental::Toolbox::LinearBarcodeDetectionResult>,
            readLinearCodes,
            const Zivid::Frame2D &,
            frame2d,
            const Zivid::Experimental::Toolbox::LinearBarcodeFormatFilter &,
            filter)
        ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(
            std::vector<Zivid::Experimental::Toolbox::MatrixBarcodeDetectionResult>,
            readMatrixCodes,
            const Zivid::Frame2D &,
            frame2d,
            const Zivid::Experimental::Toolbox::MatrixBarcodeFormatFilter &,
            filter)
    };

    void wrapEnum(pybind11::enum_<Zivid::Experimental::Toolbox::LinearBarcodeFormat> pyEnum);
    void wrapEnum(pybind11::enum_<Zivid::Experimental::Toolbox::MatrixBarcodeFormat> pyEnum);
    void wrapClass(pybind11::class_<Zivid::Experimental::Toolbox::LinearBarcodeDetectionResult> pyClass);
    void wrapClass(pybind11::class_<Zivid::Experimental::Toolbox::MatrixBarcodeDetectionResult> pyClass);
    void wrapClass(pybind11::class_<ReleasableBarcodeDetector> pyClass);
} // namespace ZividPython
