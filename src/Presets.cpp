#include <ZividPython/Presets.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{

    void wrapClass(pybind11::class_<Zivid::Presets::Preset> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Preset::name).def("settings", &Zivid::Presets::Preset::settings);
    }

    void wrapClass(pybind11::class_<Zivid::Presets::Preset2D> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Preset2D::name).def("settings", &Zivid::Presets::Preset2D::settings);
    }

    void wrapClass(pybind11::class_<Zivid::Presets::Category> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Category::name).def("presets", &Zivid::Presets::Category::presets);
    }

    void wrapClass(pybind11::class_<Zivid::Presets::Category2D> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Category2D::name).def("presets", &Zivid::Presets::Category2D::presets);
    }

    namespace Presets
    {
        void wrapAsSubmodule(pybind11::module &dest)
        {
            using namespace Zivid::Presets;

            ZIVID_PYTHON_WRAP_CLASS(dest, Preset);
            ZIVID_PYTHON_WRAP_CLASS(dest, Preset2D);
            ZIVID_PYTHON_WRAP_CLASS(dest, Category);
            ZIVID_PYTHON_WRAP_CLASS(dest, Category2D);

            dest.def("categories", &categories);
            dest.def("categories2d", &categories2D);
        }
    } // namespace Presets

} // namespace ZividPython