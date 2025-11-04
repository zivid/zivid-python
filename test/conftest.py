import numbers
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zivid


def _testdata_dir():
    return (Path(__file__).parent.parent / "test" / "test_data").resolve()


@pytest.fixture(name="datamodel_yml_dir")
def datamodel_yml_dir_fixture():
    return _testdata_dir() / "datamodels"


@pytest.fixture(name="application", scope="module")
def application_fixture():
    with zivid.Application() as app:
        yield app


@pytest.fixture(name="file_camera_file", scope="module")
def file_camera_file_fixture():
    return _testdata_dir() / "FileCameraZivid2M70.zfc"


@pytest.fixture(name="physical_camera", scope="function")
def physical_camera_fixture(application):
    with application.connect_camera() as cam:
        yield cam


@pytest.fixture(name="file_camera", scope="function")
def file_camera_fixture(application, file_camera_file):
    with application.create_file_camera(file_camera_file) as file_cam:
        yield file_cam


@pytest.fixture(name="shared_file_camera", scope="module")
def shared_file_camera_fixture(application, file_camera_file):
    with application.create_file_camera(file_camera_file) as file_cam:
        yield file_cam


@pytest.fixture(name="file_camera_calibration_board", scope="module")
def file_camera_calibration_board_fixture(application):
    with application.create_file_camera(_testdata_dir() / "calibration_board.zfc") as cam:
        yield cam


@pytest.fixture(name="default_settings", scope="module")
def default_settings_fixture():
    return zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])


@pytest.fixture(name="default_settings_2d", scope="module")
def default_settings_2d_fixture():
    return zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])


@pytest.fixture(name="frame_file", scope="module")
def frame_file_fixture(shared_file_camera, default_settings):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_frame.zdf"
        with shared_file_camera.capture(default_settings) as frame:
            frame.save(file_path)
        yield file_path


@pytest.fixture(name="checkerboard_frames", scope="module")
def checkerboard_frames_fixture(
    application,  # pylint: disable=unused-argument
):
    frames = [zivid.Frame(file_path) for file_path in sorted(_testdata_dir().glob("checkerboard_*.zdf"))]
    assert len(frames) == 3
    yield frames
    for frame in frames:
        frame.release()


@pytest.fixture(name="calibration_board_frame", scope="module")
def calibration_board_frame_fixture(
    application,  # pylint: disable=unused-argument
):
    with zivid.Frame(_testdata_dir() / "ZVD-CB01.zdf") as frame:
        yield frame


@pytest.fixture(name="calibration_board_and_aruco_markers_frame", scope="module")
def calibration_board_and_aruco_markers_frame_fixture(
    application,  # pylint: disable=unused-argument
):
    with zivid.Frame(_testdata_dir() / "handeye" / "eth" / "img01.zdf") as frame:
        yield frame


@pytest.fixture(name="barcodes_frame", scope="module")
def barcodes_frame_fixture(
    application,  # pylint: disable=unused-argument
):
    with zivid.Frame(_testdata_dir() / "barcodes.zdf") as frame:
        yield frame


@pytest.fixture(name="multicamera_transforms", scope="module")
def multicamera_transforms_fixture():
    transforms = [
        np.loadtxt(str(path), delimiter=",") for path in sorted(_testdata_dir().glob("multicamera_transform_*.csv"))
    ]
    return transforms


@pytest.fixture(name="handeye_eth_frames", scope="module")
def handeye_eth_frames_fixture(
    application,  # pylint: disable=unused-argument
):
    path = _testdata_dir() / "handeye" / "eth"
    frames = [zivid.Frame(file_path) for file_path in sorted(path.glob("*.zdf"))]
    yield frames
    for frame in frames:
        frame.release()


@pytest.fixture(name="frame", scope="function")
def frame_fixture(shared_file_camera, default_settings):
    with shared_file_camera.capture(default_settings) as frame:
        yield frame


@pytest.fixture(name="frame_2d", scope="function")
def frame_2d_fixture(shared_file_camera, default_settings_2d):
    with shared_file_camera.capture(default_settings_2d) as frame2d:
        yield frame2d


@pytest.fixture(name="point_cloud", scope="function")
def point_cloud_fixture(frame):
    with frame.point_cloud() as point_cloud:
        yield point_cloud


@pytest.fixture(name="handeye_eth_poses", scope="function")
def handeye_eth_poses_fixture():
    path = _testdata_dir() / "handeye" / "eth"
    transforms = [np.loadtxt(str(path), delimiter=",") for path in sorted(path.glob("pos*.csv"))]
    return transforms


@pytest.fixture(name="handeye_eth_transform", scope="function")
def handeye_eth_transform_fixture():
    path = _testdata_dir() / "handeye" / "eth" / "eth_transform.csv"
    return np.loadtxt(str(path), delimiter=",")


@pytest.fixture(name="handeye_marker_eth_transform", scope="function")
def handeye_marker_eth_transform_fixture():
    path = _testdata_dir() / "handeye" / "eth" / "eth_transform_marker.csv"
    return np.loadtxt(str(path), delimiter=",")


@pytest.fixture(name="handeye_eth_low_dof_markers_transform", scope="function")
def handeye_eth_low_dof_markers_transform_fixture():
    path = _testdata_dir() / "handeye" / "eth" / "low_dof" / "eth_low_dof_transform_marker.csv"
    return np.loadtxt(str(path), delimiter=",")


@pytest.fixture(name="handeye_eth_low_dof_transform", scope="function")
def handeye_eth_low_dof_transform_fixture():
    path = _testdata_dir() / "handeye" / "eth" / "low_dof" / "eth_low_dof_transform.csv"
    return np.loadtxt(str(path), delimiter=",")


@pytest.fixture(name="handeye_eth_low_dof_fixed_calibration_board_pose", scope="function")
def handeye_eth_low_dof_fixed_calibration_board_pose_fixture():
    path = _testdata_dir() / "handeye" / "eth" / "low_dof" / "eth_low_dof_fixed_calibration_board_pose.csv"
    return np.loadtxt(str(path), delimiter=",")


@pytest.fixture(name="handeye_eth_low_dof_fixed_markers_id_position_list", scope="function")
def handeye_eth_low_dof_fixed_markers_id_position_list_fixture():
    markers_id_position_list = []
    for i in range(1, 5):
        path = _testdata_dir() / "handeye" / "eth" / "low_dof" / f"eth_low_dof_fixed_marker_id_{i}.csv"
        pose = np.loadtxt(str(path), delimiter=",")
        position = pose[:3, 3]
        markers_id_position_list.append((i, position))
    return markers_id_position_list


@pytest.fixture(name="markers_2d_corners", scope="function")
def markers_2d_corners_fixture():
    path = _testdata_dir() / "marker_detection"
    corners = {
        int(file.stem.split("_")[-1]): np.loadtxt(file, delimiter=",")
        for file in sorted(path.glob("expected_2d_corners_*.csv"))
    }
    return corners


@pytest.fixture(name="markers_3d_corners", scope="function")
def markers_3d_corners_fixture():
    path = _testdata_dir() / "marker_detection"
    corners = {
        int(file.stem.split("_")[-1]): np.loadtxt(file, delimiter=",")
        for file in sorted(path.glob("expected_3d_corners_*.csv"))
    }
    return corners


@pytest.fixture(name="markers_poses", scope="function")
def markers_poses_fixture():
    path = _testdata_dir() / "marker_detection"
    poses = {
        int(file.stem.split("_")[-1]): np.loadtxt(file, delimiter=",")
        for file in sorted(path.glob("expected_poses_*.csv"))
    }
    return poses


@pytest.fixture(name="transform", scope="function")
def transform_fixture():
    return np.array(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0, 20.0],
            [0.0, 1.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture(name="file_camera_info", scope="function")
def file_camera_info_fixture(shared_file_camera):
    yield shared_file_camera.info


@pytest.fixture(name="frame_info", scope="function")
def frame_info_fixture(frame):
    yield frame.info


@pytest.helpers.register
def set_attribute_tester(settings_instance, member, value, expected_data_type):
    if not hasattr(settings_instance, member):
        raise RuntimeError(
            "Settings instance {settings_instance} does not have the member: {member}".format(
                settings_instance=settings_instance, member=member
            )
        )
    setattr(settings_instance, member, value)
    assert getattr(settings_instance, member) == value
    assert isinstance(getattr(settings_instance, member), expected_data_type)

    class DummyClass:  # pylint: disable=too-few-public-methods
        pass

    with pytest.raises(TypeError):
        setattr(settings_instance, member, DummyClass())
    if expected_data_type in (int, float, numbers.Real):
        with pytest.raises(IndexError):
            setattr(settings_instance, member, 999999999)
        with pytest.raises(IndexError):
            setattr(settings_instance, member, -999999999)
    elif expected_data_type is bool:
        pass
    elif expected_data_type is list:
        with pytest.raises(TypeError):
            setattr(settings_instance, member, 1)
        with pytest.raises(TypeError):
            setattr(settings_instance, member, 1.0)
        with pytest.raises(TypeError):
            setattr(settings_instance, member, True)

        # Setter for list can also take a numpy array, but this should not change the getter's type
        setattr(settings_instance, member, np.array(value))
        assert isinstance(getattr(settings_instance, member), expected_data_type)
    else:
        with pytest.raises(TypeError):
            setattr(settings_instance, member, True)


@pytest.helpers.register
def equality_tester(settings_type, value_collection_1, value_collection_2):
    instance_1 = settings_type(*value_collection_1)
    instance_2 = settings_type(*value_collection_1)
    assert instance_1 == instance_2

    instance_3 = settings_type(*value_collection_2)
    assert instance_1 != instance_3
    assert instance_3 != instance_2


@pytest.helpers.register
def check_handeye_output(inputs, handeye_output, expected_transform):
    assert isinstance(handeye_output, zivid.calibration.HandEyeOutput)
    assert handeye_output.valid()
    assert bool(handeye_output)
    assert str(handeye_output)

    # Check returned transform
    transform_returned = handeye_output.transform()
    assert isinstance(transform_returned, np.ndarray)
    assert transform_returned.shape == (4, 4)
    np.testing.assert_allclose(transform_returned, expected_transform, rtol=1e-5)

    # Check returned residuals
    residuals_returned = handeye_output.residuals()
    assert isinstance(residuals_returned, list)
    assert len(residuals_returned) == len(inputs)
    for residual in residuals_returned:
        assert isinstance(residual, zivid.calibration.HandEyeResidual)
        assert str(residual)
        assert isinstance(residual.translation(), float)
        assert residual.translation() >= 0.0
        assert isinstance(residual.rotation(), float)
        assert residual.rotation() >= 0.0


class Cd:
    def __init__(self, new_path):
        self.new_path = new_path
        self.saved_path = None

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(str(self.new_path))

    def __exit__(self, etype, value, traceback):
        os.chdir(str(self.saved_path))


@pytest.helpers.register
def run_sample(name, working_directory=None):
    sample = (Path(__file__) / ".." / ".." / "samples" / "sample_{name}.py".format(name=name)).resolve()

    if working_directory is not None:
        with Cd(working_directory):
            subprocess.check_output(
                args=(
                    sys.executable,
                    str(sample),
                )
            )
    else:
        subprocess.check_output(
            args=(
                sys.executable,
                str(sample),
            )
        )


@pytest.fixture(
    name="color_format",
    scope="function",
    params=["rgba", "bgra", "rgba_srgb", "bgra_srgb", "srgb"],
)
def color_format_fixture(request):
    return request.param


@pytest.fixture(name="image_2d", scope="function")
def image_2d_fixture(frame_2d, color_format):
    return getattr(frame_2d, f"image_{color_format}")()
