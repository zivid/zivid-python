"""Auto generated, do not edit."""

from __future__ import annotations

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class SceneConditions:
    """A description of the ambient conditions detected by the camera."""

    class AmbientLight:
        """The ambient light detected by the camera."""

        class FlickerClassification:
            """A classification of the detected ambient light flicker, if any. The values `grid50hz` and `grid60hz` indicate that ambient light matching a 50 Hz or 60 Hz power grid was detected in the scene. In those cases it is recommended to use Ambient Light Adaptation for better point cloud quality. The value `unknownFlicker` indicates that some significant time-varying ambient light was detected, but it was not possible to determine the frequency. `otherFlicker` indicates that ambient light of a particular frequency was detected but it did not match the characteristics of a standard power grid."""

            grid50hz = "grid50hz"
            grid60hz = "grid60hz"
            noFlicker = "noFlicker"
            otherFlicker = "otherFlicker"
            unknownFlicker = "unknownFlicker"

            _valid_values = {
                "grid50hz": _zivid.SceneConditions.AmbientLight.FlickerClassification.grid50hz,
                "grid60hz": _zivid.SceneConditions.AmbientLight.FlickerClassification.grid60hz,
                "noFlicker": _zivid.SceneConditions.AmbientLight.FlickerClassification.noFlicker,
                "otherFlicker": _zivid.SceneConditions.AmbientLight.FlickerClassification.otherFlicker,
                "unknownFlicker": _zivid.SceneConditions.AmbientLight.FlickerClassification.unknownFlicker,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            flicker_classification: str = _zivid.SceneConditions.AmbientLight.FlickerClassification().value,
            flicker_frequency: float | int | None = _zivid.SceneConditions.AmbientLight.FlickerFrequency().value,
        ):

            if isinstance(flicker_classification, _zivid.SceneConditions.AmbientLight.FlickerClassification.enum):
                self._flicker_classification = _zivid.SceneConditions.AmbientLight.FlickerClassification(
                    flicker_classification
                )
            elif isinstance(flicker_classification, str):
                self._flicker_classification = _zivid.SceneConditions.AmbientLight.FlickerClassification(
                    self.FlickerClassification._valid_values[flicker_classification]
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(value_type=type(flicker_classification))
                )

            if (
                isinstance(
                    flicker_frequency,
                    (
                        float,
                        int,
                    ),
                )
                or flicker_frequency is None
            ):
                self._flicker_frequency = _zivid.SceneConditions.AmbientLight.FlickerFrequency(flicker_frequency)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(flicker_frequency)
                    )
                )

        @property
        def flicker_classification(self):
            """A classification of the detected ambient light flicker, if any. The values `grid50hz` and `grid60hz` indicate that ambient light matching a 50 Hz or 60 Hz power grid was detected in the scene. In those cases it is recommended to use Ambient Light Adaptation for better point cloud quality. The value `unknownFlicker` indicates that some significant time-varying ambient light was detected, but it was not possible to determine the frequency. `otherFlicker` indicates that ambient light of a particular frequency was detected but it did not match the characteristics of a standard power grid."""
            if self._flicker_classification.value is None:
                return None
            for key, internal_value in self.FlickerClassification._valid_values.items():
                if internal_value == self._flicker_classification.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._flicker_classification))

        @property
        def flicker_frequency(self):
            """This field contains the actual frequency unless the classification is `noFlicker` or `unknownFlicker`."""
            return self._flicker_frequency.value

        @flicker_classification.setter
        def flicker_classification(self, value):
            if isinstance(value, str):
                self._flicker_classification = _zivid.SceneConditions.AmbientLight.FlickerClassification(
                    self.FlickerClassification._valid_values[value]
                )
            elif isinstance(value, _zivid.SceneConditions.AmbientLight.FlickerClassification.enum):
                self._flicker_classification = _zivid.SceneConditions.AmbientLight.FlickerClassification(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

        @flicker_frequency.setter
        def flicker_frequency(self, value):
            if (
                isinstance(
                    value,
                    (
                        float,
                        int,
                    ),
                )
                or value is None
            ):
                self._flicker_frequency = _zivid.SceneConditions.AmbientLight.FlickerFrequency(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if (
                self._flicker_classification == other._flicker_classification
                and self._flicker_frequency == other._flicker_frequency
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_scene_conditions_ambient_light(self))

    def __init__(
        self,
        ambient_light: SceneConditions.AmbientLight | None = None,
    ):

        if ambient_light is None:
            ambient_light = self.AmbientLight()
        if not isinstance(ambient_light, self.AmbientLight):
            raise TypeError("Unsupported type: {value}".format(value=type(ambient_light)))
        self._ambient_light = ambient_light

    @property
    def ambient_light(self):
        """The ambient light detected by the camera."""
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, value):
        if not isinstance(value, self.AmbientLight):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._ambient_light = value

    @classmethod
    def load(cls, file_name):
        return _to_scene_conditions(_zivid.SceneConditions(str(file_name)))

    def save(self, file_name):
        _to_internal_scene_conditions(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_scene_conditions(_zivid.SceneConditions.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_scene_conditions(self).serialize()

    def __eq__(self, other):
        if self._ambient_light == other._ambient_light:
            return True
        return False

    def __str__(self):
        return str(_to_internal_scene_conditions(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_scene_conditions(_to_internal_scene_conditions(self))


def _to_scene_conditions_ambient_light(internal_ambient_light):
    return SceneConditions.AmbientLight(
        flicker_classification=internal_ambient_light.flicker_classification.value,
        flicker_frequency=internal_ambient_light.flicker_frequency.value,
    )


def _to_scene_conditions(internal_scene_conditions):
    return SceneConditions(
        ambient_light=_to_scene_conditions_ambient_light(internal_scene_conditions.ambient_light),
    )


def _to_internal_scene_conditions_ambient_light(ambient_light):
    internal_ambient_light = _zivid.SceneConditions.AmbientLight()

    internal_ambient_light.flicker_classification = _zivid.SceneConditions.AmbientLight.FlickerClassification(
        ambient_light._flicker_classification.value
    )
    internal_ambient_light.flicker_frequency = _zivid.SceneConditions.AmbientLight.FlickerFrequency(
        ambient_light.flicker_frequency
    )

    return internal_ambient_light


def _to_internal_scene_conditions(scene_conditions):
    internal_scene_conditions = _zivid.SceneConditions()

    internal_scene_conditions.ambient_light = _to_internal_scene_conditions_ambient_light(
        scene_conditions.ambient_light
    )
    return internal_scene_conditions
