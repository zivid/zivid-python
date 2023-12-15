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
    assert pixel_mapping_handle.row_stride == 1
    assert pixel_mapping_handle.col_stride == 1
    assert pixel_mapping_handle.row_offset == 0.0
    assert pixel_mapping_handle.col_offset == 0.0

    blue_subsample2x2_settings = Settings()
    blue_subsample2x2_settings.acquisitions.append(Settings.Acquisition())
    blue_subsample2x2_settings.sampling.pixel = Settings.Sampling.Pixel.blueSubsample2x2
    pixel_mapping_handle = pixel_mapping(
        camera=file_camera, settings=blue_subsample2x2_settings
    )
    assert pixel_mapping_handle.row_stride == 2
    assert pixel_mapping_handle.col_stride == 2
    assert pixel_mapping_handle.row_offset == 0.0
    assert pixel_mapping_handle.col_offset == 0.0

    red_subsample2x2_settings = Settings()
    red_subsample2x2_settings.acquisitions.append(Settings.Acquisition())
    red_subsample2x2_settings.sampling.pixel = Settings.Sampling.Pixel.redSubsample2x2
    pixel_mapping_handle = pixel_mapping(
        camera=file_camera, settings=red_subsample2x2_settings
    )
    assert pixel_mapping_handle.row_stride == 2
    assert pixel_mapping_handle.col_stride == 2
    assert pixel_mapping_handle.row_offset == 1.0
    assert pixel_mapping_handle.col_offset == 1.0
