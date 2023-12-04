def test_pixel_mapping(file_camera):
    from zivid.experimental.calibration import pixel_mapping
    from zivid.settings import Settings

    pixel_mapping_handle = pixel_mapping(
        camera=file_camera, settings=Settings(acquisitions=[Settings.Acquisition()])
    )
    assert isinstance(pixel_mapping_handle.row_stride, int)
    assert isinstance(pixel_mapping_handle.col_stride, int)
    assert isinstance(pixel_mapping_handle.row_offset, float)
    assert isinstance(pixel_mapping_handle.col_offset, float)
