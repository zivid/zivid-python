import numpy as np
import pytest
from zivid.calibration import Pose
from zivid.experimental.toolbox.point_cloud_registration import (
    LocalPointCloudRegistrationParameters,
    LocalPointCloudRegistrationResult,
    local_point_cloud_registration,
)
from zivid.matrix4x4 import Matrix4x4


def small_translation():
    return np.array([[1, 0, 0, 3.0], [0, 1, 0, 4.0], [0, 0, 1, 5.0], [0, 0, 0, 1]], dtype=np.float32)


def test_local_point_cloud_registration_basic_usage(point_cloud):
    upc1 = point_cloud.to_unorganized_point_cloud().voxel_downsampled(voxel_size=1.0, min_points_per_voxel=1)
    upc2 = upc1.clone().transformed(small_translation())

    result = local_point_cloud_registration(target=upc1, source=upc2)

    assert isinstance(result, LocalPointCloudRegistrationResult)
    assert isinstance(result.transform(), Pose)
    assert isinstance(result.converged(), bool)
    assert isinstance(result.source_coverage(), float)
    assert isinstance(result.root_mean_square_error(), float)
    assert isinstance(str(result), str)

    assert result.converged()
    assert result.source_coverage() > 0.99
    assert result.source_coverage() <= 1.0
    assert result.root_mean_square_error() < 1.0


def test_local_point_cloud_registration_input_arg_failure_cases(point_cloud):

    upc1 = point_cloud.to_unorganized_point_cloud().voxel_downsampled(voxel_size=1.0, min_points_per_voxel=1)
    upc2 = upc1.clone().transformed(small_translation())

    _ = local_point_cloud_registration(target=upc1, source=upc2)

    with pytest.raises(TypeError):
        local_point_cloud_registration(target=upc1, source=point_cloud)
    with pytest.raises(TypeError):
        local_point_cloud_registration(target=point_cloud, source=upc2)
    with pytest.raises(TypeError):
        local_point_cloud_registration(target=point_cloud, source=point_cloud)
    with pytest.raises(TypeError):
        local_point_cloud_registration(target=upc1, source=upc2, parameters="params")
    with pytest.raises(TypeError):
        local_point_cloud_registration(target=upc1, source=upc2, initial_transform="transform")


def test_local_point_cloud_registration_different_forms_of_initial_transform(point_cloud):
    upc1 = point_cloud.to_unorganized_point_cloud().voxel_downsampled(voxel_size=1.0, min_points_per_voxel=1)
    upc2 = upc1.clone().transformed(small_translation())

    # ndarray
    _ = local_point_cloud_registration(
        target=upc1,
        source=upc2,
        initial_transform=small_translation(),
    )

    # ndarray that is not a rigid transform
    non_rigid_transform_array = np.array(
        [[1, 10.0, 0, 3.0], [0, 1, 0, 4.0], [0, 0, 1, 5.0], [0, 0, 0, 1]], dtype=np.float32
    )
    with pytest.raises(RuntimeError) as ex:
        _ = local_point_cloud_registration(
            target=upc1,
            source=upc2,
            initial_transform=non_rigid_transform_array,
        )
    assert "Input transform is not proper Rotation+Translation affine transformation matrix" in str(ex.value)
    del ex

    # zivid.Matrix4x4
    _ = local_point_cloud_registration(
        target=upc1,
        source=upc2,
        initial_transform=Matrix4x4(small_translation()),
    )

    # zivid.Matrix4x4 that is not a rigid transform
    non_rigid_transform_matrix = Matrix4x4(non_rigid_transform_array)
    with pytest.raises(RuntimeError) as ex:
        _ = local_point_cloud_registration(
            target=upc1,
            source=upc2,
            initial_transform=non_rigid_transform_matrix,
        )
    assert "Input transform is not proper Rotation+Translation affine transformation matrix" in str(ex.value)
    del ex

    # zivid.Pose
    _ = local_point_cloud_registration(
        target=upc1,
        source=upc2,
        initial_transform=Pose(small_translation()),
    )


def test_local_point_cloud_registration_parameter_tweaking(point_cloud):
    upc1 = point_cloud.to_unorganized_point_cloud().voxel_downsampled(voxel_size=0.5, min_points_per_voxel=1)
    upc2 = upc1.clone().transformed(small_translation()).voxel_downsampled(voxel_size=2.0, min_points_per_voxel=1)

    # Converges with default parameters
    result = local_point_cloud_registration(target=upc1, source=upc2)
    assert result.converged()

    # Too few iterations to converge
    params = LocalPointCloudRegistrationParameters(max_iteration_count=5)
    result = local_point_cloud_registration(target=upc1, source=upc2, parameters=params)
    assert not result.converged()

    # Too small convergence threshold to converge in time
    params = LocalPointCloudRegistrationParameters(
        max_iteration_count=20,
        convergence_criteria=LocalPointCloudRegistrationParameters.ConvergenceCriteria(
            rmse_diff_threshold=1e-6,
            source_coverage_diff_threshold=1e-6,
        ),
    )
    result = local_point_cloud_registration(target=upc1, source=upc2, parameters=params)
    assert not result.converged()

    # Increase number of iterations to converge despite small convergence threshold
    params.max_iteration_count = 200
    result = local_point_cloud_registration(target=upc1, source=upc2, parameters=params)
    assert result.converged()
