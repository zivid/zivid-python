"""Module for experimental point cloud registration. This API may change in the future."""

import _zivid
from zivid._local_point_cloud_registration_parameters import (
    LocalPointCloudRegistrationParameters,
    _to_internal_toolbox_local_point_cloud_registration_parameters,
)
from zivid.calibration import Pose
from zivid.matrix4x4 import Matrix4x4
from zivid.unorganized_point_cloud import UnorganizedPointCloud


class LocalPointCloudRegistrationResult:
    """The result of a call to local_point_cloud_registration."""

    def __init__(self, impl):
        """Initialize LocalPointCloudRegistrationResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.toolbox.LocalPointCloudRegistrationResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.toolbox.LocalPointCloudRegistrationResult
                )
            )
        self.__impl = impl

    def transform(self):
        """The transform that must be applied to the source point cloud for it to align with the target point cloud.

        Returns:
            A rigid transform as a Pose instance
        """
        return Pose(self.__impl.transform().to_matrix())

    def converged(self):
        """A boolean indicating whether the convergence criteria were satisfied before reaching the iteration limit.

        Returns:
            A boolean indicating convergence
        """
        return self.__impl.converged()

    def source_coverage(self):
        """The fraction of points in the source that has a correspondence in the target after transformation.

        Returns:
            A float in the range [0.0, 1.0]
        """
        return self.__impl.source_coverage()

    def root_mean_square_error(self):
        """The root mean squared distance between corresponding points in the source and target after transformation.

        Returns:
            A float representing the root mean square error
        """
        return self.__impl.root_mean_square_error()

    def __str__(self):
        return str(self.__impl)


def local_point_cloud_registration(target, source, parameters=None, initial_transform=None):
    """Compute alignment transform between two point clouds.

    Given a `source` point cloud and a `target` point cloud, this function attempts to compute the transform
    that must be applied to the `source` in order to align it with the `target`. This can be used to create a
    "stitched" unorganized point cloud of an object by combining data collected from different camera angles.

    This function takes an argument `initial_transform` which is used as a starting-point for the computation
    of the transform that best aligns `source` with `target`. This initial guess is usually found from e.g.
    reference markers or robot capture pose, and this function is then used to refine the alignment. If the
    overlap of `source` and `target` is already quite good, one can pass the identity pose as `initial_transform`.

    The returned transform represents the total transform needed to align `source` with `target`, i.e. it
    includes both `initial_transform` and the refinement found by the algorithm.

    Performance is very dependent on the number of points in either point cloud. To improve performance,
    voxel downsample one or both point clouds before passing them into this function. The resulting alignment
    transform can then be applied to the non-downsampled point clouds to still obtain a dense result.

    Performance is also very dependent on `MaxCorrespondenceDistance`. To improve performance, try reducing
    this value. However, keep the value larger than the typical point-to-point distance in the point clouds,
    and larger than the expected translation error in the initial guess.

    Args:
        target:               The target UnorganizedPointCloud to align with
        source:               The source UnorganizedPointCloud to be aligned with target
        parameters:           An optional LocalPointCloudRegistrationParameters instance.
                              If not provided, default parameters are used.
        initial_transform:    Optional transform (Pose/Matrix4x4/ndarray) be used as a starting point for the algorithm.
                              If not provided, the identity transform is used.

    Returns:
        A LocalPointCloudRegistrationResult instance
    """
    if not isinstance(target, UnorganizedPointCloud):
        raise TypeError(
            "Unsupported type for argument target. Got {}, expected {}".format(type(target), UnorganizedPointCloud)
        )
    if not isinstance(source, UnorganizedPointCloud):
        raise TypeError(
            "Unsupported type for argument source. Got {}, expected {}".format(type(source), UnorganizedPointCloud)
        )

    if parameters is None:
        parameters = LocalPointCloudRegistrationParameters()
    else:
        if not isinstance(parameters, LocalPointCloudRegistrationParameters):
            raise TypeError(
                "Unsupported type for argument parameters. Got {}, expected {}".format(
                    type(parameters), LocalPointCloudRegistrationParameters
                )
            )

    if initial_transform is None:
        initial_transform_pose = Pose(Matrix4x4.identity())
    elif isinstance(initial_transform, Pose):
        initial_transform_pose = initial_transform
    else:
        initial_transform_pose = Pose(initial_transform)

    return LocalPointCloudRegistrationResult(
        _zivid.toolbox.local_point_cloud_registration(
            target=target._UnorganizedPointCloud__impl,  # pylint: disable=protected-access
            source=source._UnorganizedPointCloud__impl,  # pylint: disable=protected-access
            params=_to_internal_toolbox_local_point_cloud_registration_parameters(parameters),
            initial_transform=initial_transform_pose._Pose__impl,  # pylint: disable=protected-access
        )
    )
