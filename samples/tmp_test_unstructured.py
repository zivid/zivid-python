# %%
import zivid
import time
import open3d as o3d
import numpy as np

# %%
app = zivid.Application()


# %% Time how long it takes to get XYZ data
frame = zivid.Frame(
    "/home/aursand/dev/zivid-sdk/sdk/test-data/ReferenceScenes/ronyx/singleAcquisition/stripe_blueSubsample2x2_rgb/expectedOutputAllFiltersOn.zdf"
)
pc = frame.point_cloud()

t0 = time.time()
upc = pc.to_unorganized_point_cloud()
xyz = upc.copy_data("xyz")
print(f"Time to get unstructed data on CPU (incl conversion): {(time.time()-t0)*1000:.2f} ms")


# Transform
t0 = time.time()
upc.transform(np.eye(4))
print(f"Time to transform original: {(time.time()-t0)*1000:.2f} ms")

# Voxel downsample
t0 = time.time()
upc_downsampled = upc.voxel_downsampled(voxel_size=5.0, min_points_per_voxel=1)
print(f"Time to voxel downsample: {(time.time()-t0)*1000:.2f} ms")

t0 = time.time()
upc_downsampled.transform(np.eye(4))
print(f"Time to transform downsampled: {(time.time()-t0)*1000:.2f} ms")


# Extend
t0 = time.time()
upc_extended = upc.extended(upc_downsampled)
print(f"Time to extend: {(time.time()-t0)*1000:.2f} ms")


print(xyz.shape)
xyz_downsampled = upc_downsampled.copy_data("xyz")
xyz_extended = upc_extended.copy_data("xyz")
print(xyz.shape)
print(xyz_downsampled.shape)
print(xyz_extended.shape)

# %% Compared with the numpy way
frame = zivid.Frame(
    "/home/aursand/dev/zivid-sdk/sdk/test-data/ReferenceScenes/ronyx/singleAcquisition/stripe_blueSubsample2x2_rgb/expectedOutputAllFiltersOn.zdf"
)
pc = frame.point_cloud()

t0 = time.time()
xyz_structured = pc.copy_data("xyz")
xyz_flattened = xyz_structured.reshape(-1, 3)
xyz_alt = xyz_flattened[~np.isnan(xyz_flattened[:, 0])].copy()

print(f"{(time.time()-t0)*1000:.2f} ms")


# %%
rgba = upc.copy_data("rgba")
rgb = rgba[:, 0:3].astype(np.float32) / 255.0
pc_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
pc_open3d.colors = o3d.utility.Vector3dVector(rgb**0.5)


xyz_downsampled = upc_downsampled.copy_data("xyz")
pc_open3d_downsampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_downsampled))
pc_open3d_downsampled.paint_uniform_color((1.0, 0.0, 0.0))


# o3d.visualization.draw_geometries(
#    [pc_open3d, pc_open3d_downsampled],
#    zoom=0.7,
#    front=[0.0, 0.0, -1.0],
#    lookat=[0.0, 0.0, 600.0],
#    up=[0.0, -1.0, 0.0],
# )
