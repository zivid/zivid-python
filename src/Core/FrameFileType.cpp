#include <ZividPython/FrameFileType.h>

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::FrameFileType> pyEnum)
    {
        pyEnum.value("frame", Zivid::FrameFileType::frame).value("frame_2d", Zivid::FrameFileType::frame2D);
    }
} // namespace ZividPython
