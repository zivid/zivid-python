import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zivid
from assertions import assert_point_clouds_equal, assert_point_clouds_not_equal
from zivid import CameraInfo, CameraState, FrameInfo, Settings
from zivid.mask import Mask
from zivid.resolution import Resolution


def test_illegal_init(
    application,  # pylint: disable=unused-argument
):
    with pytest.raises(RuntimeError):
        zivid.frame.Frame("non-exisiting-file.zdf")
    with pytest.raises(TypeError):
        zivid.frame.Frame(None)
    with pytest.raises(TypeError):
        zivid.frame.Frame(12345)


def test_point_cloud(frame):
    point_cloud = frame.point_cloud()
    assert isinstance(point_cloud, zivid.PointCloud)


def test_frame_2d(frame):
    frame_2d = frame.frame_2d()
    assert frame_2d
    assert isinstance(frame_2d, zivid.Frame2D)


def test_path_init(
    application,  # pylint: disable=unused-argument
    frame_file,
):
    frame = zivid.frame.Frame(frame_file)
    assert isinstance(frame_file, Path)
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


def test_str_as_path_init(
    application,  # pylint: disable=unused-argument
    frame_file,
):
    frame = zivid.frame.Frame(str(frame_file))
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


def test_save(frame):
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "save_test.zdf"
        frame.save(save_path)
        assert save_path.exists()


def test_context_manager(
    application,  # pylint: disable=unused-argument
    frame_file,
):
    with zivid.frame.Frame(frame_file) as frame:
        frame.point_cloud()
    with pytest.raises(RuntimeError):
        frame.point_cloud()


def test_to_string(frame):
    string = str(frame)
    assert string
    assert isinstance(string, str)


def test_load(frame, frame_file):
    assert frame.load(frame_file) is None


def test_settings(frame):
    settings = frame.settings
    assert settings
    assert isinstance(settings, Settings)


def test_state(frame):
    state = frame.state
    assert state
    assert isinstance(state, CameraState)


def test_info(frame):
    info = frame.info
    assert info
    assert isinstance(info, FrameInfo)


def test_camera_info(frame):
    camera_info = frame.camera_info
    assert camera_info
    assert isinstance(camera_info, CameraInfo)


def test_release(frame):
    frame.point_cloud()
    frame.release()
    with pytest.raises(RuntimeError):
        frame.point_cloud()


def test_copy(frame, transform):
    frame_copy = copy.copy(frame)
    assert frame_copy
    assert frame_copy is not frame
    assert isinstance(frame_copy, type(frame))

    assert_point_clouds_equal(frame.point_cloud(), frame_copy.point_cloud())
    # shallow copy, point cloud transform should affect both
    frame.point_cloud().transform(transform)
    assert_point_clouds_equal(frame.point_cloud(), frame_copy.point_cloud())

    frame.release()
    assert isinstance(frame_copy.point_cloud(), zivid.PointCloud)


def test_deepcopy(frame, transform):
    frame_deepcopy = copy.deepcopy(frame)
    assert frame_deepcopy
    assert frame_deepcopy is not frame
    assert isinstance(frame_deepcopy, type(frame))

    assert_point_clouds_equal(frame.point_cloud(), frame_deepcopy.point_cloud())
    # deep copy, point cloud transform should not affect both
    frame.point_cloud().transform(transform)
    assert_point_clouds_not_equal(frame.point_cloud(), frame_deepcopy.point_cloud())

    frame.release()
    assert isinstance(frame_deepcopy.point_cloud(), zivid.PointCloud)


def test_clone(frame, transform):
    frame_clone = frame.clone()
    assert frame_clone
    assert frame_clone is not frame
    assert isinstance(frame_clone, type(frame))

    assert_point_clouds_equal(frame.point_cloud(), frame_clone.point_cloud())
    # clone, point cloud transform should not affect both
    frame.point_cloud().transform(transform)
    assert_point_clouds_not_equal(frame.point_cloud(), frame_clone.point_cloud())

    frame.release()
    assert isinstance(frame_clone.point_cloud(), zivid.PointCloud)


def test_mask_with_zivid_mask(frame):
    """Test masking with zivid.Mask object."""
    height, width = frame.point_cloud().height, frame.point_cloud().width

    # Create mask using zivid.Mask
    mask_array = np.ones((height, width), dtype=np.uint8)
    mask_array[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = False  # Only keep center region

    mask = zivid.Mask(mask_array)

    # Apply mask
    frame.mask(mask)

    # Check results
    masked_xyz = frame.point_cloud().copy_data("xyz")

    # Outside center region should be NaN
    # Top region
    top_region = masked_xyz[: height // 3, :, 0]
    assert np.all(np.isnan(top_region))

    # Bottom region
    bottom_region = masked_xyz[2 * height // 3 :, :, 0]
    assert np.all(np.isnan(bottom_region))

    # Left and right regions
    left_region = masked_xyz[:, : width // 3, 0]
    assert np.all(np.isnan(left_region))

    right_region = masked_xyz[:, 2 * width // 3 :, 0]
    assert np.all(np.isnan(right_region))


def test_masked_returns_new_instance(frame):
    """Test that masked() returns a new PointCloud instance."""
    height, width = frame.point_cloud().height, frame.point_cloud().width

    # Create mask
    mask_array = np.zeros((height, width), dtype=bool)
    mask_array[10:20, 10:20] = True

    # Get original data for comparison
    original_xyz = frame.point_cloud().copy_data("xyz").copy()

    # Apply masked() - should return new instance
    masked_point_cloud = frame.point_cloud().masked(mask_array)

    # Should be different instances
    assert masked_point_cloud is not frame.point_cloud()
    assert isinstance(masked_point_cloud, zivid.PointCloud)

    # Original should be unchanged
    current_xyz = frame.point_cloud().copy_data("xyz")
    np.testing.assert_array_equal(original_xyz, current_xyz)

    # New instance should have mask applied
    masked_xyz = masked_point_cloud.copy_data("xyz")
    masked_region = masked_xyz[10:20, 10:20, 0]
    assert np.all(np.isnan(masked_region))

    # Clean up
    masked_point_cloud.release()


def test_mask_frame_with_numpy_array(frame):
    """Test masking frame with numpy array."""
    height, width = frame.point_cloud().height, frame.point_cloud().width

    # Create mask as numpy array
    mask_array = np.ones((height, width), dtype=bool)
    mask_array[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = False  # Keep center region

    # Apply mask to frame
    frame.mask(mask_array)

    # Check results
    masked_xyz = frame.point_cloud().copy_data("xyz")

    # Outside center region should be NaN
    # Top region should be NaN
    top_region = masked_xyz[: height // 4, :, 0]
    assert np.all(np.isnan(top_region))

    # Bottom region should be NaN
    bottom_region = masked_xyz[3 * height // 4 :, :, 0]
    assert np.all(np.isnan(bottom_region))


def test_masked_frame_returns_new_instance(frame):
    """Test that frame.masked() returns a new Frame instance."""
    height, width = frame.point_cloud().height, frame.point_cloud().width

    # Create mask
    mask_array = np.zeros((height, width), dtype=bool)
    mask_array[5:15, 5:15] = True

    # Get original data for comparison
    original_xyz = frame.point_cloud().copy_data("xyz").copy()

    # Apply masked() - should return new instance
    masked_frame = frame.masked(mask_array)

    # Should be different instances
    assert masked_frame is not frame
    assert isinstance(masked_frame, zivid.Frame)

    # Original should be unchanged
    current_xyz = frame.point_cloud().copy_data("xyz")
    np.testing.assert_array_equal(original_xyz, current_xyz)

    # New instance should have mask applied
    masked_xyz = masked_frame.point_cloud().copy_data("xyz")
    masked_region = masked_xyz[5:15, 5:15, 0]
    assert np.all(np.isnan(masked_region))

    # Clean up
    masked_frame.release()


def test_mask_can_be_released():
    """Test that Mask objects can be released."""
    # Create a mask
    resolution = Resolution(100, 80)
    mask = Mask(resolution)

    # Should be able to access properties
    assert mask.width == 100
    assert mask.height == 80

    # Release the mask
    mask.release()

    # After release, should raise RuntimeError when accessing properties
    with pytest.raises(RuntimeError):
        _ = mask.width


def test_mask_context_manager():
    """Test that Mask works as context manager."""
    resolution = Resolution(50, 40)

    with Mask(resolution) as mask:
        assert mask.width == 50
        assert mask.height == 40

    # After exiting context, should raise RuntimeError
    with pytest.raises(RuntimeError):
        _ = mask.width
