#include <ZividPython/Releasable.h>
#include <ZividPython/Wrappers.h>

#include <Zivid/Visualization/Visualizer.h>

namespace ZividPython
{
    class SingletonVisualizer : public Singleton<Zivid::Visualization::Visualizer>
    {
    public:
        using Singleton::Singleton;

        ZIVID_PYTHON_FORWARD_0_ARGS(show)
        ZIVID_PYTHON_FORWARD_0_ARGS(hide)
        ZIVID_PYTHON_FORWARD_0_ARGS(run)
        ZIVID_PYTHON_FORWARD_0_ARGS(close)
        ZIVID_PYTHON_FORWARD_2_ARGS(resize, int, h, int, w)
        ZIVID_PYTHON_FORWARD_0_ARGS(resetToFit)
        ZIVID_PYTHON_FORWARD_0_ARGS(showFullScreen)
        ZIVID_PYTHON_FORWARD_0_ARGS(showMaximized)
        ZIVID_PYTHON_FORWARD_1_ARGS(setWindowTitle, const std::string, title)
        ZIVID_PYTHON_FORWARD_1_ARGS(show, const Zivid::PointCloud &, cloud)
        ZIVID_PYTHON_FORWARD_1_ARGS(show, const Zivid::UnorganizedPointCloud &, cloud)
        ZIVID_PYTHON_FORWARD_1_ARGS(show, const Zivid::Frame &, frame)
        ZIVID_PYTHON_FORWARD_1_ARGS(setColorsEnabled, bool, enabled)
        ZIVID_PYTHON_FORWARD_0_ARGS(colorsEnabled)
        ZIVID_PYTHON_FORWARD_1_ARGS(setMeshingEnabled, bool, enabled)
        ZIVID_PYTHON_FORWARD_0_ARGS(isMeshingEnabled)
        ZIVID_PYTHON_FORWARD_1_ARGS(setAxisIndicatorEnabled, bool, enabled)
        ZIVID_PYTHON_FORWARD_0_ARGS(isAxisIndicatorEnabled)
        ZIVID_PYTHON_FORWARD_0_ARGS(toString)
    };

    void wrapClass(pybind11::class_<SingletonVisualizer> pyClass);
} // namespace ZividPython
