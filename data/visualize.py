import open3d as o3d
import os


def visualize_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    filename = 'temp.ply'
    o3d.io.write_point_cloud(filename, pcd)
    pcd_load = o3d.io.read_point_cloud(filename)
    o3d.visualization.draw_geometries([pcd_load])
    os.unlink(filename)
