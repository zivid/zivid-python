#include <ZividPython/Presets.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{

    void wrapClass(pybind11::class_<Zivid::Presets::Preset> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Preset::name).def("settings", &Zivid::Presets::Preset::settings);
    }

    void wrapClass(pybind11::class_<Zivid::Presets::Category> pyClass)
    {
        pyClass.def("name", &Zivid::Presets::Category::name).def("presets", &Zivid::Presets::Category::presets);
    }

    namespace Presets
    {
        void wrapAsSubmodule(pybind11::module &dest)
        {
            using namespace Zivid::Presets;

            ZIVID_PYTHON_WRAP_CLASS(dest, Preset);
            ZIVID_PYTHON_WRAP_CLASS(dest, Category);

            dest.def("categories", &categories);
        }
    } // namespace Presets

} // namespace ZividPython