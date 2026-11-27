import pytest
import zivid

_BYTE_COLOR_FORMATS = [
    zivid.PixelFormat.RGBA,
    zivid.PixelFormat.BGRA,
    zivid.PixelFormat.RGBA_SRGB,
    zivid.PixelFormat.BGRA_SRGB,
    zivid.PixelFormat.RGB,
    zivid.PixelFormat.RGB_SRGB,
    zivid.PixelFormat.BGR,
    zivid.PixelFormat.BGR_SRGB,
]


def test_device_array_to_image(frame_2d, sdk_stream_or_queue):
    for color_format in _BYTE_COLOR_FORMATS:
        device_array = frame_2d.image_device_array(sdk_stream_or_queue, color_format)
        image = device_array.to_image(sdk_stream_or_queue)
        zivid.synchronize_stream(sdk_stream_or_queue)
        assert isinstance(image, zivid.Image)
        assert [image.height, image.width] == list(device_array.shape[:2])


def test_device_array_to_image_rejects_float_format(frame_2d, sdk_stream_or_queue):
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBAF)
    with pytest.raises(TypeError):
        device_array.to_image(sdk_stream_or_queue)


def test_device_array_to_image_rejects_wrong_stream_type(frame_2d, sdk_stream_or_queue):
    device_array = frame_2d.image_device_array(sdk_stream_or_queue, zivid.PixelFormat.RGBA)
    with pytest.raises(TypeError):
        device_array.to_image("not a stream")


@pytest.fixture(name="torch_cuda")
def torch_cuda_fixture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Test requires PyTorch with CUDA")
    return torch


def test_device_array_view_to_image(application, torch_cuda):
    compute_device = application.compute_device()
    if compute_device.backend != zivid.ComputeBackend.cuda:
        pytest.skip("Test requires the CUDA backend")
    height, width = 4, 6
    tensor = torch_cuda.zeros((height, width, 4), dtype=torch_cuda.uint8, device="cuda")
    torch_cuda.cuda.synchronize()
    stream = zivid.CUDAStreamPtr(torch_cuda.cuda.Stream().cuda_stream)
    view = compute_device.create_device_array_view_rgba(tensor.data_ptr(), width, height, stream)
    image = view.to_image(stream)
    zivid.synchronize_stream(stream)
    assert isinstance(image, zivid.Image)
    assert [image.height, image.width] == [height, width]


def test_device_array_view_to_image_rejects_float_format(application, torch_cuda):
    compute_device = application.compute_device()
    if compute_device.backend != zivid.ComputeBackend.cuda:
        pytest.skip("Test requires the CUDA backend")
    tensor = torch_cuda.zeros((4, 6, 4), dtype=torch_cuda.float32, device="cuda")
    torch_cuda.cuda.synchronize()
    stream = zivid.CUDAStreamPtr(torch_cuda.cuda.Stream().cuda_stream)
    view = compute_device.create_device_array_view_rgbaf(tensor.data_ptr(), 6, 4, stream)
    with pytest.raises(TypeError):
        view.to_image(stream)
