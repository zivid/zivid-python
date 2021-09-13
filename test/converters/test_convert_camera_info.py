def test_to_internal_camera_info_to_camera_info_modified():
    from zivid import CameraInfo
    from zivid.camera_info import _to_camera_info, _to_internal_camera_info

    modified_camera_info = CameraInfo(model_name="hello")

    converted_camera_info = _to_camera_info(
        _to_internal_camera_info(modified_camera_info)
    )
    assert modified_camera_info == converted_camera_info
    assert isinstance(converted_camera_info, CameraInfo)
    assert isinstance(modified_camera_info, CameraInfo)


def test_to_internal_camera_info_to_camera_info_default():
    from zivid import CameraInfo
    from zivid.camera_info import _to_camera_info, _to_internal_camera_info

    default_camera_info = CameraInfo()
    converted_camera_info = _to_camera_info(
        _to_internal_camera_info(default_camera_info)
    )
    assert default_camera_info == converted_camera_info
    assert isinstance(converted_camera_info, CameraInfo)
    assert isinstance(default_camera_info, CameraInfo)


#
