import open3d as o3d
import numpy as np

def visualize_results(points, segmentation_labels, instance_labels, bounding_boxes):
    """
    Visualize segmentation and bounding box results using Open3D.
    :param points: LiDAR points (Nx3).
    :param segmentation_labels: Semantic segmentation labels for each point (N).
    :param instance_labels: Instance segmentation labels for each point (N).
    :param bounding_boxes: Bounding boxes (list of Nx3 arrays).
    """
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color points by segmentation labels
    colors = np.random.rand(len(np.unique(segmentation_labels)), 3)
    pcd.colors = o3d.utility.Vector3dVector(colors[segmentation_labels])

    # Create bounding boxes
    bbox_list = []
    for bbox in bounding_boxes:
        corners = o3d.utility.Vector3dVector(bbox)
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [6, 7], [7, 4]]
        bbox = o3d.geometry.LineSet(points=corners, lines=o3d.utility.Vector2iVector(lines))
        bbox.paint_uniform_color([1, 0, 0])  # Red for bounding boxes
        bbox_list.append(bbox)

    # Visualize
    o3d.visualization.draw_geometries([pcd] + bbox_list)
