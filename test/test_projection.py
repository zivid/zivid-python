import numpy as np
import pytest
from zivid import Frame2D, Settings, Settings2D
from zivid.projection import ProjectedImage, pixels_from_3d_points, projector_resolution, show_image_bgra


@pytest.mark.physical_camera
def test_projector_resolution(physical_camera):
    res = projector_resolution(camera=physical_camera)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], int)
    assert isinstance(res[1], int)


@pytest.mark.physical_camera
def test_show_image_bgra(physical_camera):
    # Exception if wrong image resolution
    bgra_wrong_resolution = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        show_image_bgra(camera=physical_camera, image_bgra=bgra_wrong_resolution)

    # Make image with correct resolution
    res = projector_resolution(camera=physical_camera)
    bgra = 255 * np.ones((res[0], res[1], 4))

    # Project (with context manager)
    with show_image_bgra(camera=physical_camera, image_bgra=bgra) as projected_image:
        assert isinstance(projected_image, ProjectedImage)
        assert projected_image.active()
        assert str(projected_image) == "Active: true"

    # Project (without context manager)
    projected_image = show_image_bgra(camera=physical_camera, image_bgra=bgra)
    assert isinstance(projected_image, ProjectedImage)
    assert projected_image.active()
    assert str(projected_image) == "Active: true"
    projected_image.stop()
    assert not projected_image.active()
    assert str(projected_image) == "Active: false"


@pytest.mark.physical_camera
def test_capture_while_projecting(physical_camera):
    res = projector_resolution(camera=physical_camera)
    bgra = 255 * np.ones((res[0], res[1], 4))

    def is_r_series_camera(model):
        return "R" in model

    with show_image_bgra(camera=physical_camera, image_bgra=bgra) as projected_image:
        settings2d = Settings2D()
        settings2d.acquisitions.append(Settings2D.Acquisition(brightness=0.0))
        settings2d.sampling.color = (
            Settings2D.Sampling.Color.grayscale
            if is_r_series_camera(physical_camera.info.model)
            else Settings2D.Sampling.Color.rgb
        )

        with projected_image.capture_2d(settings2d) as frame_2d:
            assert isinstance(frame_2d, Frame2D)

        settings = Settings(color=settings2d)
        with projected_image.capture_2d(settings) as frame_2d:
            assert isinstance(frame_2d, Frame2D)

        with pytest.raises(RuntimeError):
            projected_image.capture_2d(Settings())

        with pytest.raises(RuntimeError):
            projected_image.capture_2d(Settings2D(acquisitions=[Settings2D.Acquisition(brightness=1.0)]))


@pytest.mark.physical_camera
def test_3d_to_projector_pixels(physical_camera):
    points = [[0.0, 0.0, 1000.0], [10.0, -10.0, 1200.0]]
    projector_coords = pixels_from_3d_points(camera=physical_camera, points=points)
    assert isinstance(projector_coords, list)
    assert len(projector_coords) == len(points)
    for coord in projector_coords:
        assert isinstance(coord, list)
        assert len(coord) == 2
        assert isinstance(coord[0], float)
        assert isinstance(coord[1], float)

    # Numpy array should also be valid input
    pixels_from_3d_points(camera=physical_camera, points=np.array(points))
