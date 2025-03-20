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
upc = pc.to_unstructured_point_cloud()
xyz = upc.copy_data("xyz")
print(f"{(time.time()-t0)*1000:.2f} ms")

print(xyz.shape)


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

print(xyz_alt.shape)

# %%
rgba = upc.copy_data("rgba")


# %%

rgb = rgba[:, 0:3].astype(np.float32) / 255.0
pc_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
pc_open3d.colors = o3d.utility.Vector3dVector(rgb**0.5)

o3d.visualization.draw_geometries(
    [pc_open3d],
    zoom=0.7,
    front=[0.0, 0.0, -1.0],
    lookat=[0.0, 0.0, 600.0],
    up=[0.0, -1.0, 0.0],
)
