from pathlib import Path
import tempfile
import numpy
import numpy.testing
import pytest
import zivid

ZIVID_MATRIX_SAVE_LOAD_TOLERANCE_DECIMAL = 5


def to_float32_1d(arr):
    for i, element in enumerate(arr):
        arr[i] = float(numpy.float32(element))
    return arr


def to_float32_2d(arr):
    for row in arr:
        to_float32_1d(row)
    return arr


def assert_all_equal_flat(matrix: zivid.Matrix4x4, arr2) -> None:
    assert isinstance(matrix, zivid.Matrix4x4)
    assert len(arr2) == 16
    for element1, element2 in zip(matrix, arr2):
        numpy.testing.assert_almost_equal(
            element1, element2, ZIVID_MATRIX_SAVE_LOAD_TOLERANCE_DECIMAL
        )


def assert_all_equal_2d(matrix: zivid.Matrix4x4, arr2) -> None:
    assert isinstance(matrix, zivid.Matrix4x4)
    assert not isinstance(arr2, zivid.Matrix4x4)
    assert len(arr2) == 4
    for row in range(4):
        assert len(arr2[row]) == 4
        for col in range(4):
            numpy.testing.assert_almost_equal(
                matrix[row, col],
                arr2[row][col],
                ZIVID_MATRIX_SAVE_LOAD_TOLERANCE_DECIMAL,
            )


def sample_2d_list() -> list:
    return to_float32_2d(
        [
            [-8.3, 6.75, -2, 5.7],
            [-0.24, 1.49, 3.5, -4.25],
            [52, 0.98, 970.6, 75000],
            [-64.3, 15.4, -84.4, -13.4],
        ]
    )


def sample_1d_list() -> list:
    return to_float32_1d(
        [
            3.8,
            0.64,
            -9.55,
            63226,
            -0.3445,
            9.5,
            4.004,
            0.115,
            9999,
            -34,
            14,
            66.5,
            87.4,
            1.3,
            36.2,
            93.12,
        ]
    )


def test_default_init():
    assert_all_equal_flat(zivid.Matrix4x4(), [0.0] * 16)


def test_flat_array_init():
    assert_all_equal_flat(zivid.Matrix4x4(sample_1d_list()), sample_1d_list())
    assert_all_equal_flat(
        zivid.Matrix4x4(numpy.array(sample_1d_list())), numpy.array(sample_1d_list())
    )

    with pytest.raises(TypeError):
        zivid.Matrix4x4(range(42))

    with pytest.raises(TypeError):
        zivid.Matrix4x4(range(0))


def test_4x4_array_init():
    assert_all_equal_2d(zivid.Matrix4x4(sample_2d_list()), sample_2d_list())
    assert_all_equal_2d(
        zivid.Matrix4x4(sample_2d_list()), numpy.array(sample_2d_list())
    )

    with pytest.raises(TypeError):
        zivid.Matrix4x4([range(4), range(4, 9), range(9, 13), range(13, 17)])

    with pytest.raises(TypeError):
        zivid.Matrix4x4([[], range(4, 8), range(8, 12), range(12, 16)])

    with pytest.raises(TypeError):
        zivid.Matrix4x4([[]] * 4)


def test_getitem():
    matrix = zivid.Matrix4x4(sample_2d_list())

    for i in range(4):
        for j in range(4):
            assert matrix[i, j] == sample_2d_list()[i][j]

    for i in range(-4, 0):
        for j in range(-4, 0):
            assert matrix[i, j] == sample_2d_list()[i][j]

    with pytest.raises(TypeError):
        assert matrix[0] == 0

    with pytest.raises(TypeError):
        assert matrix[0, 0, 0] == 0

    with pytest.raises(TypeError):
        assert matrix[1.4, 1.0] == 0

    with pytest.raises(TypeError):
        assert matrix["0", "0"] == 0

    with pytest.raises(IndexError):
        assert matrix[1000, 0] == 0

    with pytest.raises(IndexError):
        assert matrix[0, 1000] == 0

    with pytest.raises(IndexError):
        assert matrix[0, -1000] == 0


def test_setitem():
    matrix = zivid.Matrix4x4()

    for i in range(4):
        for j in range(4):
            matrix[i, j] = sample_2d_list()[i][j]
            assert matrix[i, j] == sample_2d_list()[i][j]

    assert_all_equal_2d(matrix, sample_2d_list())

    for i in range(-4, 0):
        for j in range(-4, 0):
            matrix[i, j] = sample_2d_list()[i][j]
            assert matrix[i, j] == sample_2d_list()[i][j]

    assert_all_equal_2d(matrix, sample_2d_list())

    with pytest.raises(TypeError):
        matrix[0] = 0

    with pytest.raises(TypeError):
        matrix[0, 0, 0] = 0

    with pytest.raises(TypeError):
        matrix[1.4, 1.0] = 0

    with pytest.raises(TypeError):
        matrix["0", "0"] = 0

    with pytest.raises(TypeError):
        matrix[0, 0] = "42"

    with pytest.raises(TypeError):
        matrix[0, 0] = 10**1000

    with pytest.raises(IndexError):
        matrix[1000, 0] = 0

    with pytest.raises(IndexError):
        matrix[0, 1000] = 0

    with pytest.raises(IndexError):
        matrix[0, -1000] = 0


def test_rows_cols():
    assert zivid.Matrix4x4.rows == 4
    assert zivid.Matrix4x4.cols == 4
    assert zivid.Matrix4x4().rows == 4
    assert zivid.Matrix4x4().cols == 4

    with pytest.raises(AttributeError):
        zivid.Matrix4x4.rows = 3

    with pytest.raises(AttributeError):
        zivid.Matrix4x4.cols = 3


def test_inverse():
    def invertible_matrix():
        return [
            [1, 1, 1, -1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [-1, 1, 1, 1],
        ]

    def non_invertible_matrix():
        return [1] * 16

    matrix = zivid.Matrix4x4(invertible_matrix())

    assert_all_equal_2d(
        matrix.inverse(),
        [
            [0.25, 0.25, 0.25, -0.25],
            [0.25, 0.25, -0.25, 0.25],
            [0.25, -0.25, 0.25, 0.25],
            [-0.25, 0.25, 0.25, 0.25],
        ],
    )

    assert_all_equal_2d(matrix, invertible_matrix())

    matrix = zivid.Matrix4x4(non_invertible_matrix())

    with pytest.raises(RuntimeError):
        matrix.inverse()

    assert_all_equal_flat(matrix, non_invertible_matrix())


def test_buffer_protocol():
    numpy_array = numpy.array(zivid.Matrix4x4(sample_2d_list()))
    assert (numpy_array == sample_2d_list()).all()


def test_to_string():
    matrix = zivid.Matrix4x4(sample_2d_list())

    assert str(matrix) == (
        "[ [-8.300000,  6.750000, -2.000000,  5.700000], \n"
        "  [-0.240000,  1.490000,  3.500000, -4.250000], \n"
        "  [ 52.000000,  0.980000,  970.599976,  75000.000000], \n"
        "  [-64.300003,  15.400000, -84.400002, -13.400000] ]"
    )


def test_save():
    with tempfile.TemporaryDirectory() as tmpdir, zivid.Application() as _:
        file = Path(tmpdir) / "matrix_saved.yml"
        matrix = zivid.Matrix4x4(
            to_float32_2d(
                [
                    [0.5, 1.34, -234, -3.43],
                    [-4.31, 5343, 6.34, 7.12],
                    [-8, -9, -10.3, 11.2],
                    [1.2, 13.5, 1.4, 0.15],
                ]
            )
        )
        assert not file.exists()

        matrix.save(file)

        assert file.exists()

        expected_content = (
            "FloatMatrix:\n"
            "  Data: [\n"
            "    [0.5, 1.34, -234, -3.43],\n"
            "    [-4.31, 5343, 6.34, 7.12],\n"
            "    [-8, -9, -10.3, 11.2],\n"
            "    [1.2, 13.5, 1.4, 0.15]]\n"
        )

        with open(file, "r", encoding="utf8") as file:
            assert expected_content in file.read()


def test_load():
    with tempfile.TemporaryDirectory() as tmpdir, zivid.Application() as _:
        file = Path(tmpdir) / "matrix_saved.yml"
        zivid.Matrix4x4(sample_2d_list()).save(file)
        matrix = zivid.Matrix4x4()
        matrix.load(file)
        assert_all_equal_2d(matrix, sample_2d_list())


def test_file_init():
    with tempfile.TemporaryDirectory() as tmpdir, zivid.Application() as _:
        file = Path(tmpdir) / "matrix_saved.yml"
        zivid.Matrix4x4(sample_2d_list()).save(file)
        assert_all_equal_2d(zivid.Matrix4x4(file), sample_2d_list())


def test_implicit_convert_to_numpy():
    sample = to_float32_2d(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0, 20.0],
            [0.0, 1.0, 0.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose1 = zivid.calibration.Pose(zivid.Matrix4x4(sample))
    pose2 = zivid.calibration.Pose(numpy.array(sample))

    assert_all_equal_2d(zivid.Matrix4x4(pose1.to_matrix()), pose2.to_matrix())


def test_identity():
    identity = zivid.Matrix4x4.identity()
    assert numpy.array_equal(identity, numpy.identity(4, dtype=numpy.float32))
