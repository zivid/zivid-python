#pragma once

#include <Zivid/Frame.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableFrame2D.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableFrame : public Releasable<Zivid::Frame>
    {
    public:
        using Releasable::Releasable;

        void release() override
        {
            if(m_frame2D != nullptr)
            {
                m_frame2D->release();
                m_frame2D = nullptr;
            }
            Releasable::release();
        }

        ZIVID_PYTHON_FORWARD_1_ARGS(save, const std::string &, fileName)
        ZIVID_PYTHON_FORWARD_1_ARGS(load, const std::string &, fileName)

        std::shared_ptr<ReleasablePointCloud> pointCloud()
        {
            pybind11::gil_scoped_release gilLock;

            if(m_pointCloud == nullptr || m_pointCloud->isReleased())
            {
                m_pointCloud = std::make_shared<ReleasablePointCloud>(impl().pointCloud());
            }

            return m_pointCloud;
        }

        std::optional<std::shared_ptr<ReleasableFrame2D>> frame2D()
        {
            pybind11::gil_scoped_release gilLock;

            auto frame = impl().frame2D();
            if(!frame.has_value())
            {
                return std::nullopt;
            }

            if(m_frame2D == nullptr || m_frame2D->isReleased())
            {
                m_frame2D = std::make_shared<ReleasableFrame2D>(std::move(frame.value()));
            }

            return m_frame2D;
        }
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(info)
        ZIVID_PYTHON_FORWARD_0_ARGS(cameraInfo)

    private:
        std::shared_ptr<ReleasablePointCloud> m_pointCloud{ nullptr };
        std::shared_ptr<ReleasableFrame2D> m_frame2D{ nullptr };
    };

    void wrapClass(pybind11::class_<ReleasableFrame> pyClass);
} // namespace ZividPython
