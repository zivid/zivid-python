"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class LocalPointCloudRegistrationParameters:

    class ConvergenceCriteria:

        def __init__(
            self,
            rmse_diff_threshold=_zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.RMSEDiffThreshold().value,
            source_coverage_diff_threshold=_zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.SourceCoverageDiffThreshold().value,
        ):

            if isinstance(
                rmse_diff_threshold,
                (
                    float,
                    int,
                ),
            ):
                self._rmse_diff_threshold = (
                    _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.RMSEDiffThreshold(
                        rmse_diff_threshold
                    )
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(rmse_diff_threshold)
                    )
                )

            if isinstance(
                source_coverage_diff_threshold,
                (
                    float,
                    int,
                ),
            ):
                self._source_coverage_diff_threshold = _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.SourceCoverageDiffThreshold(
                    source_coverage_diff_threshold
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(source_coverage_diff_threshold)
                    )
                )

        @property
        def rmse_diff_threshold(self):
            return self._rmse_diff_threshold.value

        @property
        def source_coverage_diff_threshold(self):
            return self._source_coverage_diff_threshold.value

        @rmse_diff_threshold.setter
        def rmse_diff_threshold(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._rmse_diff_threshold = (
                    _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.RMSEDiffThreshold(value)
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
                )

        @source_coverage_diff_threshold.setter
        def source_coverage_diff_threshold(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._source_coverage_diff_threshold = _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.SourceCoverageDiffThreshold(
                    value
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if (
                self._rmse_diff_threshold == other._rmse_diff_threshold
                and self._source_coverage_diff_threshold == other._source_coverage_diff_threshold
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_toolbox_local_point_cloud_registration_parameters_convergence_criteria(self))

    def __init__(
        self,
        max_correspondence_distance=_zivid.toolbox.LocalPointCloudRegistrationParameters.MaxCorrespondenceDistance().value,
        max_iteration_count=_zivid.toolbox.LocalPointCloudRegistrationParameters.MaxIterationCount().value,
        convergence_criteria=None,
    ):

        if isinstance(
            max_correspondence_distance,
            (
                float,
                int,
            ),
        ):
            self._max_correspondence_distance = (
                _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxCorrespondenceDistance(
                    max_correspondence_distance
                )
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (float, int,), got {value_type}".format(
                    value_type=type(max_correspondence_distance)
                )
            )

        if isinstance(max_iteration_count, (int,)):
            self._max_iteration_count = _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxIterationCount(
                max_iteration_count
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (int,), got {value_type}".format(value_type=type(max_iteration_count))
            )

        if convergence_criteria is None:
            convergence_criteria = self.ConvergenceCriteria()
        if not isinstance(convergence_criteria, self.ConvergenceCriteria):
            raise TypeError("Unsupported type: {value}".format(value=type(convergence_criteria)))
        self._convergence_criteria = convergence_criteria

    @property
    def max_correspondence_distance(self):
        return self._max_correspondence_distance.value

    @property
    def max_iteration_count(self):
        return self._max_iteration_count.value

    @property
    def convergence_criteria(self):
        return self._convergence_criteria

    @max_correspondence_distance.setter
    def max_correspondence_distance(self, value):
        if isinstance(
            value,
            (
                float,
                int,
            ),
        ):
            self._max_correspondence_distance = (
                _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxCorrespondenceDistance(value)
            )
        else:
            raise TypeError(
                "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
            )

    @max_iteration_count.setter
    def max_iteration_count(self, value):
        if isinstance(value, (int,)):
            self._max_iteration_count = _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxIterationCount(value)
        else:
            raise TypeError("Unsupported type, expected: int, got {value_type}".format(value_type=type(value)))

    @convergence_criteria.setter
    def convergence_criteria(self, value):
        if not isinstance(value, self.ConvergenceCriteria):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._convergence_criteria = value

    @classmethod
    def load(cls, file_name):
        return _to_toolbox_local_point_cloud_registration_parameters(
            _zivid.toolbox.LocalPointCloudRegistrationParameters(str(file_name))
        )

    def save(self, file_name):
        _to_internal_toolbox_local_point_cloud_registration_parameters(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_toolbox_local_point_cloud_registration_parameters(
            _zivid.toolbox.LocalPointCloudRegistrationParameters.from_serialized(str(value))
        )

    def serialize(self):
        return _to_internal_toolbox_local_point_cloud_registration_parameters(self).serialize()

    def __eq__(self, other):
        if (
            self._max_correspondence_distance == other._max_correspondence_distance
            and self._max_iteration_count == other._max_iteration_count
            and self._convergence_criteria == other._convergence_criteria
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_toolbox_local_point_cloud_registration_parameters(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_toolbox_local_point_cloud_registration_parameters(
            _to_internal_toolbox_local_point_cloud_registration_parameters(self)
        )


def _to_toolbox_local_point_cloud_registration_parameters_convergence_criteria(internal_convergence_criteria):
    return LocalPointCloudRegistrationParameters.ConvergenceCriteria(
        rmse_diff_threshold=internal_convergence_criteria.rmse_diff_threshold.value,
        source_coverage_diff_threshold=internal_convergence_criteria.source_coverage_diff_threshold.value,
    )


def _to_toolbox_local_point_cloud_registration_parameters(internal_local_point_cloud_registration_parameters):
    return LocalPointCloudRegistrationParameters(
        convergence_criteria=_to_toolbox_local_point_cloud_registration_parameters_convergence_criteria(
            internal_local_point_cloud_registration_parameters.convergence_criteria
        ),
        max_correspondence_distance=internal_local_point_cloud_registration_parameters.max_correspondence_distance.value,
        max_iteration_count=internal_local_point_cloud_registration_parameters.max_iteration_count.value,
    )


def _to_internal_toolbox_local_point_cloud_registration_parameters_convergence_criteria(convergence_criteria):
    internal_convergence_criteria = _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria()

    internal_convergence_criteria.rmse_diff_threshold = (
        _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.RMSEDiffThreshold(
            convergence_criteria.rmse_diff_threshold
        )
    )
    internal_convergence_criteria.source_coverage_diff_threshold = (
        _zivid.toolbox.LocalPointCloudRegistrationParameters.ConvergenceCriteria.SourceCoverageDiffThreshold(
            convergence_criteria.source_coverage_diff_threshold
        )
    )

    return internal_convergence_criteria


def _to_internal_toolbox_local_point_cloud_registration_parameters(local_point_cloud_registration_parameters):
    internal_local_point_cloud_registration_parameters = _zivid.toolbox.LocalPointCloudRegistrationParameters()

    internal_local_point_cloud_registration_parameters.max_correspondence_distance = (
        _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxCorrespondenceDistance(
            local_point_cloud_registration_parameters.max_correspondence_distance
        )
    )
    internal_local_point_cloud_registration_parameters.max_iteration_count = (
        _zivid.toolbox.LocalPointCloudRegistrationParameters.MaxIterationCount(
            local_point_cloud_registration_parameters.max_iteration_count
        )
    )

    internal_local_point_cloud_registration_parameters.convergence_criteria = (
        _to_internal_toolbox_local_point_cloud_registration_parameters_convergence_criteria(
            local_point_cloud_registration_parameters.convergence_criteria
        )
    )
    return internal_local_point_cloud_registration_parameters
