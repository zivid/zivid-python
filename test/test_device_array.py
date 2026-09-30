import ctypes
import gc
import logging

import numpy as np
import pytest
import zivid

logger = logging.getLogger(__name__)

_KDLCUDA = 2
_KDLCPU = 1


_PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
_PyCapsule_GetName.restype = ctypes.c_char_p
_PyCapsule_GetName.argtypes = [ctypes.py_object]


def _capsule_name(capsule):
    return _PyCapsule_GetName(capsule).decode()


_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _versioned_capsule_version(capsule):
    """Read the (major, minor) DLPackVersion stamped at the start of a DLManagedTensorVersioned."""
    pointer = _PyCapsule_GetPointer(capsule, b"dltensor_versioned")
    major, minor = (ctypes.c_uint32 * 2).from_address(pointer)
    return (major, minor)


_FOUR_CHANNEL_BYTE_FORMATS = [
    zivid.PixelFormat.RGBA,
    zivid.PixelFormat.BGRA,
    zivid.PixelFormat.RGBA_SRGB,
    zivid.PixelFormat.BGRA_SRGB,
]

_THREE_CHANNEL_FORMATS = [
    zivid.PixelFormat.RGB,
    zivid.PixelFormat.RGB_SRGB,
    zivid.PixelFormat.BGR,
    zivid.PixelFormat.BGR_SRGB,
]


@pytest.fixture(name="cuda_compute_device")
def cuda_compute_device_fixture(application):
    if application.compute_device().backend != zivid.ComputeBackend.cuda:
        pytest.skip("Test requires the CUDA compute backend")
    return application.compute_device()


@pytest.fixture(name="torch_cuda")
def torch_cuda_fixture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Test requires PyTorch with CUDA")
    return torch


def test_image(frame_2d, sdk_stream_or_queue):
    for color_format in _FOUR_CHANNEL_BYTE_FORMATS + [zivid.PixelFormat.RGBAF]:
        device_array = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
        assert isinstance(device_array, zivid.device_array.DeviceArray)
        assert device_array.shape[2] == 4

    for color_format in _THREE_CHANNEL_FORMATS:
        device_array = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
        assert isinstance(device_array, zivid.device_array.DeviceArray)
        assert device_array.shape[2] == 3


def test_image_copy_to_host_all_formats(frame_2d, sdk_stream_or_queue):
    for color_format in _FOUR_CHANNEL_BYTE_FORMATS + _THREE_CHANNEL_FORMATS:
        device_array = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
        host_array = device_array.copy_to_host_organized_array(sdk_stream_or_queue)
        zivid.synchronize_stream(sdk_stream_or_queue)
        assert isinstance(host_array, np.ndarray)
        assert host_array.ndim == 3
        assert host_array.shape[0] == device_array.shape[0]
        assert host_array.shape[1] == device_array.shape[1]
        assert host_array.shape[2] == device_array.shape[2]
        assert host_array.strides[0] == device_array.strides_in_bytes[0]
        assert host_array.strides[1] == device_array.strides_in_bytes[1]
        assert host_array.strides[2] == device_array.strides_in_bytes[2]


def test_point_cloud_copy_to_host_organized(point_cloud, sdk_stream_or_queue):
    device_arrays = [
        getattr(point_cloud, method_name)(sdk_stream_or_queue)
        for method_name in [
            "device_points_xyz",
            "device_points_xyzw",
            "device_points_z",
            "device_snrs",
            "device_normals_xyz",
        ]
    ]
    device_arrays += [
        point_cloud.device_image(sdk_stream_or_queue, color_format) for color_format in _FOUR_CHANNEL_BYTE_FORMATS
    ]
    for device_array in device_arrays:
        assert isinstance(device_array, zivid.device_array.DeviceArray)
        host_array = device_array.copy_to_host_organized_array(sdk_stream_or_queue)
        zivid.synchronize_stream(sdk_stream_or_queue)
        assert isinstance(host_array, np.ndarray)
        assert host_array.ndim >= 2
        assert host_array.shape[0] == device_array.shape[0]
        assert host_array.shape[1] == device_array.shape[1]
        if host_array.ndim > 2:
            assert host_array.shape[2] == device_array.shape[2]
        assert host_array.strides[0] == device_array.strides_in_bytes[0]
        assert host_array.strides[1] == device_array.strides_in_bytes[1]
        if host_array.ndim > 2:
            assert host_array.strides[2] == device_array.strides_in_bytes[2]


def test_point_cloud_copy_to_host_unorganized(point_cloud, sdk_stream_or_queue):
    upc = point_cloud.to_unorganized_point_cloud()
    device_arrays = [
        getattr(upc, method_name)(sdk_stream_or_queue) for method_name in ["device_points_xyz", "device_points_xyzw"]
    ]
    device_arrays += [
        upc.device_colors(sdk_stream_or_queue, color_format) for color_format in _FOUR_CHANNEL_BYTE_FORMATS
    ]
    for device_array in device_arrays:
        assert isinstance(device_array, zivid.device_array.DeviceArray)
        host_array = device_array.copy_to_host_unorganized_array(sdk_stream_or_queue)
        zivid.synchronize_stream(sdk_stream_or_queue)
        assert isinstance(host_array, np.ndarray)
        assert host_array.ndim == 2
        assert host_array.shape[0] == device_array.shape[0]
        assert host_array.shape[1] == device_array.shape[1]
        assert host_array.strides[0] == device_array.strides_in_bytes[0]
        assert host_array.strides[1] == device_array.strides_in_bytes[1]

    device_snrs = upc.device_snrs(sdk_stream_or_queue)
    assert isinstance(device_snrs, zivid.device_array.DeviceArray)
    host_snrs = device_snrs.copy_to_host_unorganized_array(sdk_stream_or_queue)
    zivid.synchronize_stream(sdk_stream_or_queue)
    assert isinstance(host_snrs, np.ndarray)
    assert host_snrs.ndim == 1
    assert host_snrs.shape[0] == device_snrs.shape[0]
    assert host_snrs.strides[0] == device_snrs.strides_in_bytes[0]


def test_organized_device_array_rejects_copy_to_host_unorganized(point_cloud, sdk_stream_or_queue):
    device_arrays = [
        getattr(point_cloud, method_name)(sdk_stream_or_queue)
        for method_name in [
            "device_points_xyz",
            "device_points_xyzw",
            "device_points_z",
            "device_snrs",
            "device_normals_xyz",
        ]
    ]
    device_arrays += [
        point_cloud.device_image(sdk_stream_or_queue, color_format) for color_format in _FOUR_CHANNEL_BYTE_FORMATS
    ]
    for device_array in device_arrays:
        with pytest.raises(RuntimeError):
            device_array.copy_to_host_unorganized_array(sdk_stream_or_queue)


@pytest.mark.parametrize("color_format", _FOUR_CHANNEL_BYTE_FORMATS + _THREE_CHANNEL_FORMATS)
def test_image_device_array_fill_matches_allocate(
    frame_2d, cuda_compute_device, sdk_stream_or_queue, torch_cuda, color_format
):
    _ = cuda_compute_device
    reference = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
    reference_host = reference.copy_to_host_organized_array(sdk_stream_or_queue)
    zivid.synchronize_stream(sdk_stream_or_queue)

    height, width, channels = reference.shape
    user_stream = zivid.CUDAStreamPtr(torch_cuda.cuda.current_stream().cuda_stream)
    destination = torch_cuda.zeros((height, width, channels), dtype=torch_cuda.uint8, device="cuda")
    destination_view = zivid.create_device_array_view(destination, user_stream, color_format)
    frame_2d.image_device_array_fill(user_stream, destination_view)
    zivid.synchronize_stream(user_stream)

    np.testing.assert_array_equal(reference_host, destination.cpu().numpy())


def test_image_device_array_fill_rgbaf_writes_image(frame_2d, cuda_compute_device, sdk_stream_or_queue, torch_cuda):
    _ = cuda_compute_device
    height, width = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF).shape[:2]
    user_stream = zivid.CUDAStreamPtr(torch_cuda.cuda.current_stream().cuda_stream)
    destination = torch_cuda.zeros((height, width, 4), dtype=torch_cuda.float32, device="cuda")
    destination_view = zivid.create_device_array_view(destination, user_stream, zivid.PixelFormat.RGBAF)
    frame_2d.image_device_array_fill(user_stream, destination_view)
    zivid.synchronize_stream(user_stream)
    assert bool(destination.any())


def test_image_device_array_rejects_non_pixel_format(frame_2d, sdk_stream_or_queue):
    with pytest.raises(TypeError):
        frame_2d.image_device_array(sdk_stream_or_queue, "rgba")


def test_three_channel_device_array_to_image(frame_2d, sdk_stream_or_queue):
    """to_image() on 3-channel device arrays returns a host-side Image with matching pixels."""
    cases = [
        (zivid.PixelFormat.RGB, "image_rgb"),
        (zivid.PixelFormat.RGB_SRGB, "image_rgb_srgb"),
        (zivid.PixelFormat.BGR, "image_bgr"),
        (zivid.PixelFormat.BGR_SRGB, "image_bgr_srgb"),
    ]
    for color_format, host_method in cases:
        device_array = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
        image_via_device = zivid.Image(
            device_array._impl.to_image(sdk_stream_or_queue)  # pylint: disable=protected-access
        )
        zivid.synchronize_stream(sdk_stream_or_queue)
        image_via_host = getattr(frame_2d, host_method)()
        assert image_via_device.height == image_via_host.height, color_format
        assert image_via_device.width == image_via_host.width, color_format
        np.testing.assert_array_equal(image_via_device.copy_data(), image_via_host.copy_data())


def test_three_channel_image_save_load_roundtrip(frame_2d, tmp_path):
    """Image<ColorRGB[_SRGB]> and Image<ColorBGR[_SRGB]> save and load round-trip via PNG."""
    cases = [
        ("image_rgb", "rgb"),
        ("image_rgb_srgb", "rgb_srgb"),
        ("image_bgr", "bgr"),
        ("image_bgr_srgb", "bgr_srgb"),
    ]
    for host_method, color_format in cases:
        image = getattr(frame_2d, host_method)()
        path = tmp_path / f"{host_method}.png"
        image.save(path)
        loaded = zivid.Image.load(path, color_format=color_format)
        np.testing.assert_array_equal(image.copy_data(), loaded.copy_data())


def test_unorganized_device_array_rejects_copy_to_host_organized(point_cloud, sdk_stream_or_queue):
    upc = point_cloud.to_unorganized_point_cloud()
    device_arrays = [
        getattr(upc, method_name)(sdk_stream_or_queue)
        for method_name in ["device_points_xyz", "device_points_xyzw", "device_snrs"]
    ]
    device_arrays += [
        upc.device_colors(sdk_stream_or_queue, color_format) for color_format in _FOUR_CHANNEL_BYTE_FORMATS
    ]
    for device_array in device_arrays:
        with pytest.raises(RuntimeError):
            device_array.copy_to_host_organized_array(sdk_stream_or_queue)


def test_point_cloud_device_image_rgbaf(point_cloud, sdk_stream_or_queue):
    device_array = point_cloud.device_image(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    assert isinstance(device_array, zivid.device_array.DeviceArray)
    assert device_array.shape[2] == 4
    assert device_array.is_valid


def test_unorganized_device_colors_rgbaf(point_cloud, sdk_stream_or_queue):
    upc = point_cloud.to_unorganized_point_cloud()
    device_array = upc.device_colors(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    assert isinstance(device_array, zivid.device_array.DeviceArray)
    assert device_array.shape[-1] == 4
    assert device_array.is_valid


def test_unorganized_device_colors_rejects_rgb(point_cloud, sdk_stream_or_queue):
    upc = point_cloud.to_unorganized_point_cloud()
    with pytest.raises(ValueError):
        upc.device_colors(sdk_stream_or_queue, zivid.PixelFormat.RGB)


def test_unorganized_device_colors_rgbaf_is_unquantized_rgba(
    point_cloud, cuda_compute_device, sdk_stream_or_queue, torch_cuda
):
    _ = cuda_compute_device
    upc = point_cloud.to_unorganized_point_cloud()
    rgba_host = upc.device_colors(sdk_stream_or_queue, zivid.PixelFormat.RGBA).copy_to_host_unorganized_array(
        sdk_stream_or_queue
    )
    rgbaf = upc.device_colors(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    zivid.synchronize_stream(sdk_stream_or_queue)
    rgbaf_host = torch_cuda.as_tensor(rgbaf, device="cuda").cpu().numpy()

    scaled = np.squeeze(rgbaf_host) * 255.0
    quantized = np.squeeze(rgba_host).astype(np.float32)
    assert np.all(scaled >= -1e-3)
    assert np.all(scaled <= 255.0 + 1e-3)
    np.testing.assert_allclose(scaled, quantized, atol=0.5 + 1e-3)


def test_device_array_dlpack_protocol_without_a_consumer(frame_2d, cuda_compute_device, sdk_stream_or_queue):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)

    device_type, device_id = device_array.__dlpack_device__()
    assert device_type == _KDLCUDA
    assert device_id >= 0

    legacy_capsule = device_array.__dlpack__()
    assert type(legacy_capsule).__name__ == "PyCapsule"
    assert _capsule_name(legacy_capsule) == "dltensor"

    legacy_by_version = device_array.__dlpack__(stream=None, max_version=(0, 8), dl_device=None, copy=None)
    assert _capsule_name(legacy_by_version) == "dltensor"

    versioned_capsule = device_array.__dlpack__(max_version=(1, 0))
    assert _capsule_name(versioned_capsule) == "dltensor_versioned"
    assert versioned_capsule is not legacy_capsule

    newer_consumer = device_array.__dlpack__(max_version=(2, 5))
    assert _capsule_name(newer_consumer) == "dltensor_versioned"

    # Per the array API producer protocol the capsule carries our own max version, also when the
    # consumer's minor is lower than ours; only an older major falls back to the legacy tensor.
    ours = _versioned_capsule_version(versioned_capsule)
    for consumer_max_version in [(1, 0), (1, 1), (1, 99), (2, 5), (9, 9)]:
        capsule = device_array.__dlpack__(max_version=consumer_max_version)
        assert _capsule_name(capsule) == "dltensor_versioned"
        assert _versioned_capsule_version(capsule) == ours


def test_device_array_dlpack_honours_our_own_device(frame_2d, cuda_compute_device, sdk_stream_or_queue):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    assert _capsule_name(device_array.__dlpack__(dl_device=device_array.__dlpack_device__())) == "dltensor"


def test_device_array_dlpack_rejects_another_device(frame_2d, cuda_compute_device, sdk_stream_or_queue):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    _, ordinal = device_array.__dlpack_device__()
    for dl_device in [(_KDLCPU, 0), (_KDLCUDA, ordinal + 1)]:
        with pytest.raises(BufferError):
            device_array.__dlpack__(dl_device=dl_device)


def test_device_array_dlpack_copy_semantics(frame_2d, cuda_compute_device, sdk_stream_or_queue):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)

    # copy=True would have to hand out independent memory, which we cannot produce.
    with pytest.raises(BufferError):
        device_array.__dlpack__(copy=True)

    # copy=False must never raise, since we never copy.
    assert _capsule_name(device_array.__dlpack__(copy=False)) == "dltensor"
    assert _capsule_name(device_array.__dlpack__(copy=None)) == "dltensor"


def test_device_array_dlpack_device_reports_cuda(frame_2d, cuda_compute_device, sdk_stream_or_queue, torch_cuda):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    device_type, device_id = device_array.__dlpack_device__()
    assert device_type == _KDLCUDA
    assert device_id == torch_cuda.cuda.current_device()


def test_device_array_dlpack_roundtrip_is_zero_copy(frame_2d, cuda_compute_device, sdk_stream_or_queue, torch_cuda):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    tensor = torch_cuda.from_dlpack(device_array)
    assert tensor.data_ptr() == device_array.device_pointer()
    assert tuple(tensor.shape) == tuple(device_array.shape)
    assert tensor.dtype == torch_cuda.float32


def test_device_array_dlpack_keeps_buffer_alive(frame_2d, cuda_compute_device, sdk_stream_or_queue, torch_cuda):
    _ = cuda_compute_device
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    device_pointer = device_array.device_pointer()
    tensor = torch_cuda.from_dlpack(device_array)
    zivid.synchronize_stream(sdk_stream_or_queue)
    del device_array
    gc.collect()
    assert tensor.data_ptr() == device_pointer
    assert bool(torch_cuda.isfinite(tensor).all())


def test_unorganized_device_array_dlpack_roundtrip(point_cloud, cuda_compute_device, sdk_stream_or_queue, torch_cuda):
    _ = cuda_compute_device
    upc = point_cloud.to_unorganized_point_cloud()
    device_array = upc.device_points_xyz(sdk_stream_or_queue)
    tensor = torch_cuda.from_dlpack(device_array)
    zivid.synchronize_stream(sdk_stream_or_queue)
    assert tensor.data_ptr() == device_array.device_pointer()
    assert tensor.dtype == torch_cuda.float32
    assert bool(torch_cuda.isfinite(tensor).all())
