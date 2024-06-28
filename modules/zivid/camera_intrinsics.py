"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class CameraIntrinsics:

    class CameraMatrix:

        def __init__(
            self,
            cx=_zivid.CameraIntrinsics.CameraMatrix.CX().value,
            cy=_zivid.CameraIntrinsics.CameraMatrix.CY().value,
            fx=_zivid.CameraIntrinsics.CameraMatrix.FX().value,
            fy=_zivid.CameraIntrinsics.CameraMatrix.FY().value,
        ):

            if isinstance(
                cx,
                (
                    float,
                    int,
                ),
            ):
                self._cx = _zivid.CameraIntrinsics.CameraMatrix.CX(cx)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(cx)
                    )
                )

            if isinstance(
                cy,
                (
                    float,
                    int,
                ),
            ):
                self._cy = _zivid.CameraIntrinsics.CameraMatrix.CY(cy)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(cy)
                    )
                )

            if isinstance(
                fx,
                (
                    float,
                    int,
                ),
            ):
                self._fx = _zivid.CameraIntrinsics.CameraMatrix.FX(fx)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(fx)
                    )
                )

            if isinstance(
                fy,
                (
                    float,
                    int,
                ),
            ):
                self._fy = _zivid.CameraIntrinsics.CameraMatrix.FY(fy)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(fy)
                    )
                )

        @property
        def cx(self):
            return self._cx.value

        @property
        def cy(self):
            return self._cy.value

        @property
        def fx(self):
            return self._fx.value

        @property
        def fy(self):
            return self._fy.value

        @cx.setter
        def cx(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._cx = _zivid.CameraIntrinsics.CameraMatrix.CX(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @cy.setter
        def cy(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._cy = _zivid.CameraIntrinsics.CameraMatrix.CY(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @fx.setter
        def fx(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._fx = _zivid.CameraIntrinsics.CameraMatrix.FX(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @fy.setter
        def fy(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._fy = _zivid.CameraIntrinsics.CameraMatrix.FY(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if (
                self._cx == other._cx
                and self._cy == other._cy
                and self._fx == other._fx
                and self._fy == other._fy
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_intrinsics_camera_matrix(self))

    class Distortion:

        def __init__(
            self,
            k1=_zivid.CameraIntrinsics.Distortion.K1().value,
            k2=_zivid.CameraIntrinsics.Distortion.K2().value,
            k3=_zivid.CameraIntrinsics.Distortion.K3().value,
            p1=_zivid.CameraIntrinsics.Distortion.P1().value,
            p2=_zivid.CameraIntrinsics.Distortion.P2().value,
        ):

            if isinstance(
                k1,
                (
                    float,
                    int,
                ),
            ):
                self._k1 = _zivid.CameraIntrinsics.Distortion.K1(k1)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(k1)
                    )
                )

            if isinstance(
                k2,
                (
                    float,
                    int,
                ),
            ):
                self._k2 = _zivid.CameraIntrinsics.Distortion.K2(k2)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(k2)
                    )
                )

            if isinstance(
                k3,
                (
                    float,
                    int,
                ),
            ):
                self._k3 = _zivid.CameraIntrinsics.Distortion.K3(k3)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(k3)
                    )
                )

            if isinstance(
                p1,
                (
                    float,
                    int,
                ),
            ):
                self._p1 = _zivid.CameraIntrinsics.Distortion.P1(p1)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(p1)
                    )
                )

            if isinstance(
                p2,
                (
                    float,
                    int,
                ),
            ):
                self._p2 = _zivid.CameraIntrinsics.Distortion.P2(p2)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(p2)
                    )
                )

        @property
        def k1(self):
            return self._k1.value

        @property
        def k2(self):
            return self._k2.value

        @property
        def k3(self):
            return self._k3.value

        @property
        def p1(self):
            return self._p1.value

        @property
        def p2(self):
            return self._p2.value

        @k1.setter
        def k1(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._k1 = _zivid.CameraIntrinsics.Distortion.K1(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @k2.setter
        def k2(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._k2 = _zivid.CameraIntrinsics.Distortion.K2(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @k3.setter
        def k3(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._k3 = _zivid.CameraIntrinsics.Distortion.K3(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @p1.setter
        def p1(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._p1 = _zivid.CameraIntrinsics.Distortion.P1(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @p2.setter
        def p2(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._p2 = _zivid.CameraIntrinsics.Distortion.P2(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if (
                self._k1 == other._k1
                and self._k2 == other._k2
                and self._k3 == other._k3
                and self._p1 == other._p1
                and self._p2 == other._p2
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_intrinsics_distortion(self))

    def __init__(
        self,
        camera_matrix=None,
        distortion=None,
    ):

        if camera_matrix is None:
            camera_matrix = self.CameraMatrix()
        if not isinstance(camera_matrix, self.CameraMatrix):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(camera_matrix))
            )
        self._camera_matrix = camera_matrix

        if distortion is None:
            distortion = self.Distortion()
        if not isinstance(distortion, self.Distortion):
            raise TypeError("Unsupported type: {value}".format(value=type(distortion)))
        self._distortion = distortion

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @property
    def distortion(self):
        return self._distortion

    @camera_matrix.setter
    def camera_matrix(self, value):
        if not isinstance(value, self.CameraMatrix):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._camera_matrix = value

    @distortion.setter
    def distortion(self, value):
        if not isinstance(value, self.Distortion):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._distortion = value

    @classmethod
    def load(cls, file_name):
        return _to_camera_intrinsics(_zivid.CameraIntrinsics(str(file_name)))

    def save(self, file_name):
        _to_internal_camera_intrinsics(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_camera_intrinsics(
            _zivid.CameraIntrinsics.from_serialized(str(value))
        )

    def serialize(self):
        return _to_internal_camera_intrinsics(self).serialize()

    def __eq__(self, other):
        if (
            self._camera_matrix == other._camera_matrix
            and self._distortion == other._distortion
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_camera_intrinsics(self))


def _to_camera_intrinsics_camera_matrix(internal_camera_matrix):
    return CameraIntrinsics.CameraMatrix(
        cx=internal_camera_matrix.cx.value,
        cy=internal_camera_matrix.cy.value,
        fx=internal_camera_matrix.fx.value,
        fy=internal_camera_matrix.fy.value,
    )


def _to_camera_intrinsics_distortion(internal_distortion):
    return CameraIntrinsics.Distortion(
        k1=internal_distortion.k1.value,
        k2=internal_distortion.k2.value,
        k3=internal_distortion.k3.value,
        p1=internal_distortion.p1.value,
        p2=internal_distortion.p2.value,
    )


def _to_camera_intrinsics(internal_camera_intrinsics):
    return CameraIntrinsics(
        camera_matrix=_to_camera_intrinsics_camera_matrix(
            internal_camera_intrinsics.camera_matrix
        ),
        distortion=_to_camera_intrinsics_distortion(
            internal_camera_intrinsics.distortion
        ),
    )


def _to_internal_camera_intrinsics_camera_matrix(camera_matrix):
    internal_camera_matrix = _zivid.CameraIntrinsics.CameraMatrix()

    internal_camera_matrix.cx = _zivid.CameraIntrinsics.CameraMatrix.CX(
        camera_matrix.cx
    )
    internal_camera_matrix.cy = _zivid.CameraIntrinsics.CameraMatrix.CY(
        camera_matrix.cy
    )
    internal_camera_matrix.fx = _zivid.CameraIntrinsics.CameraMatrix.FX(
        camera_matrix.fx
    )
    internal_camera_matrix.fy = _zivid.CameraIntrinsics.CameraMatrix.FY(
        camera_matrix.fy
    )

    return internal_camera_matrix


def _to_internal_camera_intrinsics_distortion(distortion):
    internal_distortion = _zivid.CameraIntrinsics.Distortion()

    internal_distortion.k1 = _zivid.CameraIntrinsics.Distortion.K1(distortion.k1)
    internal_distortion.k2 = _zivid.CameraIntrinsics.Distortion.K2(distortion.k2)
    internal_distortion.k3 = _zivid.CameraIntrinsics.Distortion.K3(distortion.k3)
    internal_distortion.p1 = _zivid.CameraIntrinsics.Distortion.P1(distortion.p1)
    internal_distortion.p2 = _zivid.CameraIntrinsics.Distortion.P2(distortion.p2)

    return internal_distortion


def _to_internal_camera_intrinsics(camera_intrinsics):
    internal_camera_intrinsics = _zivid.CameraIntrinsics()

    internal_camera_intrinsics.camera_matrix = (
        _to_internal_camera_intrinsics_camera_matrix(camera_intrinsics.camera_matrix)
    )
    internal_camera_intrinsics.distortion = _to_internal_camera_intrinsics_distortion(
        camera_intrinsics.distortion
    )
    return internal_camera_intrinsics
