# pylint: disable=import-outside-toplevel


def test_to_internal_camera_info_to_camera_info_modified():
    from zivid import CameraInfo
    from zivid._camera_info_converter import to_camera_info, to_internal_camera_info

    modified_camera_info = CameraInfo(model_name="hello")

    converted_camera_info = to_camera_info(
        to_internal_camera_info(modified_camera_info)
    )
    assert modified_camera_info == converted_camera_info
    assert isinstance(converted_camera_info, CameraInfo)
    assert isinstance(modified_camera_info, CameraInfo)


def test_to_internal_camera_info_to_camera_info_default():
    from zivid import CameraInfo
    from zivid._camera_info_converter import to_camera_info, to_internal_camera_info

    default_camera_info = CameraInfo()
    converted_camera_info = to_camera_info(to_internal_camera_info(default_camera_info))
    assert default_camera_info == converted_camera_info
    assert isinstance(converted_camera_info, CameraInfo)
    assert isinstance(default_camera_info, CameraInfo)


#
