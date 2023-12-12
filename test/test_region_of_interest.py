import numpy as np


def test_region_of_interest_depth(shared_file_camera):
    import zivid

    z_min = 650.0
    z_max = 800.0

    # Make sure default capture has points outside the range we want
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    with shared_file_camera.capture(settings) as frame_default:
        z_default = frame_default.point_cloud().copy_data("z")
    assert np.nanmin(z_default) < z_min
    assert np.nanmax(z_default) > z_max

    # Then use depth filter to limit the z-range
    settings.region_of_interest.depth.enabled = True
    settings.region_of_interest.depth.range = [z_min, z_max]
    with shared_file_camera.capture(settings) as frame_roi:
        z_roi = frame_roi.point_cloud().copy_data("z")

    assert np.nanmin(z_roi) >= z_min
    assert np.nanmax(z_roi) <= z_max


def test_region_of_interest_box(shared_file_camera):
    import zivid

    x_min = 0.0
    x_max = 150.0
    y_min = -125.0
    y_max = 0.0

    # Make sure default capture has points outside the range we want
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    with shared_file_camera.capture(settings) as frame_default:
        xyz_default = frame_default.point_cloud().copy_data("xyz")
    assert np.nanmin(xyz_default[:, :, 0]) < x_min
    assert np.nanmax(xyz_default[:, :, 0]) > x_max
    assert np.nanmin(xyz_default[:, :, 1]) < y_min
    assert np.nanmax(xyz_default[:, :, 1]) > y_max

    # Then use box filter to limit region
    settings.region_of_interest.box.enabled = True
    settings.region_of_interest.box.extents = [-500.0, 500.0]
    settings.region_of_interest.box.point_o = [x_min, y_max, 500.0]
    settings.region_of_interest.box.point_a = [x_max, y_max, 500.0]
    settings.region_of_interest.box.point_b = [x_min, y_min, 500.0]
    with shared_file_camera.capture(settings) as frame_roi:
        xyz_roi = frame_roi.point_cloud().copy_data("xyz")
    assert np.nanmin(xyz_roi[:, :, 0]) >= x_min
    assert np.nanmax(xyz_roi[:, :, 0]) <= x_max
    assert np.nanmin(xyz_roi[:, :, 1]) >= y_min
    assert np.nanmax(xyz_roi[:, :, 1]) <= y_max
