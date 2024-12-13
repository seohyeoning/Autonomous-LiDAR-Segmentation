import os
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from tqdm import tqdm  # tqdm 라이브러리 추가
import re  # 정규 표현식 라이브러리 추가


def extract_frame_id(file_name):
    """
    Extract the first frame ID from a file name (e.g., "0000000002_0000000385.ply").
    :param file_name: File name (e.g., "0000000002_0000000385.ply").
    :return: Integer frame ID (e.g., 2).
    """
    match = re.match(r'(\d+)', file_name)
    if match:
        return int(match.group(1))  # 첫 번째 숫자를 추출
    return None

def find_matching_ply(bin_file, ply_files):
    """
    Find a .ply file that matches the given .bin file based on frame ID.
    :param bin_file: Name of the .bin file (e.g., "0000000002.bin").
    :param ply_files: List of .ply file names.
    :return: Matching .ply file name or None.
    """
    bin_frame_id = extract_frame_id(bin_file)
    for ply_file in ply_files:
        ply_frame_id = extract_frame_id(ply_file)
        if ply_frame_id == bin_frame_id:
            return ply_file  # 매칭된 .ply 파일 반환
    return None  # 매칭되지 않는 경우 None 반환


def load_point_cloud(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # x, y, z 좌표만 사용

def load_labels(ply_path):
    if not os.path.exists(ply_path):
        return None  # 해당 라벨 파일이 없는 경우 처리
    pcd = o3d.io.read_point_cloud(ply_path)
    labels = np.asarray(pcd.colors)  # 라벨 정보가 색상으로 저장되어 있음
    return labels

def parse_bounding_boxes(xml_path):
    if not os.path.exists(xml_path):
        return None  # XML 파일이 없는 경우 처리
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bounding_boxes = []
    for obj in root.findall("object"):
        vertices = np.array([
            list(map(float, vertex.text.split()))
            for vertex in obj.find("vertices")
        ])
        transform = np.array([
            list(map(float, row.text.split()))
            for row in obj.find("transform")
        ])
        bounding_boxes.append({"vertices": vertices, "transform": transform})
    return bounding_boxes

def transform_bounding_boxes(bounding_boxes):
    transformed_boxes = []
    for box in bounding_boxes:
        vertices = box["vertices"]
        transform = box["transform"]
        ones = np.ones((vertices.shape[0], 1))
        vertices_homogeneous = np.hstack((vertices, ones))
        world_vertices = (transform @ vertices_homogeneous.T).T
        transformed_boxes.append(world_vertices[:, :3])
    return transformed_boxes

def voxelization(points, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid

def voxelgrid_to_numpy(voxel_grid):
    """
    Convert Open3D VoxelGrid to Numpy array.
    :param voxel_grid: Open3D VoxelGrid object.
    :return: Numpy array containing voxel centers.
    """
    voxel_centers = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    return voxel_centers


def preprocess_data(raw_dir, semantic_dir, bbox_dir, output_dir, voxel_size=0.05):
    """
    Preprocess raw LiDAR data, semantic labels, and bounding boxes, and save preprocessed data.
    """
    sequences = os.listdir(raw_dir)
    for seq in tqdm(sequences, desc="Processing sequences"):
        seq_raw_dir = os.path.join(raw_dir, seq, "velodyne_points", "data")
        seq_semantic_dir = os.path.join(semantic_dir, "train", seq, "static")
        seq_bbox_path = os.path.join(bbox_dir, "train", f"{seq}.xml")
        output_seq_dir = os.path.join(output_dir, seq)

        os.makedirs(output_seq_dir, exist_ok=True)

        bin_files = sorted(os.listdir(seq_raw_dir))
        bounding_boxes = parse_bounding_boxes(seq_bbox_path)

        for bin_file in tqdm(bin_files, desc=f"Processing frames in {seq}", leave=False):
            bin_path = os.path.join(seq_raw_dir, bin_file)
            ply_path = os.path.join(seq_semantic_dir, bin_file.replace(".bin", ".ply"))

            # Load LiDAR points
            points = load_point_cloud(bin_path)

            # Load labels (handle missing labels gracefully)
            labels = load_labels(ply_path)
            if labels is None:
                print(f"[WARNING] Missing labels for {bin_file}. Proceeding without labels.")
                labels = np.zeros((points.shape[0], 3))  # Placeholder for missing labels

            # Skip if bounding boxes are missing
            if bounding_boxes is None:
                print(f"[WARNING] Missing bounding boxes for sequence {seq}. Skipping frame {bin_file}.")
                continue

            # Perform voxelization
            voxel_grid = voxelization(points, voxel_size)

            # Convert VoxelGrid to Numpy
            voxel_centers = voxelgrid_to_numpy(voxel_grid)

            # Transform bounding boxes
            transformed_boxes = transform_bounding_boxes(bounding_boxes)

            # Save preprocessed data
            output_path = os.path.join(output_seq_dir, bin_file.replace(".bin", ".npz"))
            np.savez(output_path, voxels=voxel_centers, labels=labels, bounding_boxes=transformed_boxes)
            print(f"Saved file: {output_path}")


def preprocess_data(raw_dir, semantic_dir, bbox_dir, output_dir, voxel_size=0.05):
    sequences = os.listdir(raw_dir)

    # 전체 시퀀스 진행률
    for seq in tqdm(sequences, desc="Processing sequences"):
        seq_raw_dir = os.path.join(raw_dir, seq, "velodyne_points", "data")
        seq_semantic_dir = os.path.join(semantic_dir, "train", seq, "static")
        seq_bbox_path = os.path.join(bbox_dir, "train", f"{seq}.xml")
        output_seq_dir = os.path.join(output_dir, seq)

        os.makedirs(output_seq_dir, exist_ok=True)

        bin_files = sorted(os.listdir(seq_raw_dir))
        ply_files = sorted(os.listdir(seq_semantic_dir))  # .ply 파일 목록 로드
        bounding_boxes = parse_bounding_boxes(seq_bbox_path)

        # 시퀀스 내 프레임 진행률
        for bin_file in tqdm(bin_files, desc=f"Processing frames in {seq}", leave=False):
            bin_path = os.path.join(seq_raw_dir, bin_file)

            # Find matching .ply file
            matching_ply = find_matching_ply(bin_file, ply_files)
            if matching_ply is None:
                print(f"[WARNING] No matching .ply file found for {bin_file}. Skipping...")
                continue

            ply_path = os.path.join(seq_semantic_dir, matching_ply)

            # Debugging: Print matching file names
            print(f"Matching .bin: {bin_file} -> .ply: {matching_ply}")

            # Load LiDAR points
            points = load_point_cloud(bin_path)

            # Load labels
            labels = load_labels(ply_path)
            if labels is None:
                print(f"[WARNING] Missing labels for {bin_file}. Proceeding without labels.")
                labels = np.zeros((points.shape[0], 3))  # Placeholder for missing labels

            # Perform voxelization
            voxel_grid = voxelization(points, voxel_size)
            voxel_centers = voxelgrid_to_numpy(voxel_grid)  # VoxelGrid -> Numpy 변환
            
            # Transform bounding boxes
            transformed_boxes = transform_bounding_boxes(bounding_boxes)

            # Save preprocessed data
            output_path = os.path.join(output_seq_dir, bin_file.replace(".bin", ".npz"))
            np.savez(output_path, voxels=voxel_centers, labels=labels, bounding_boxes=transformed_boxes)
            print(f"Saved file: {output_path}")


# 사용 예시
raw_dir = "C:/Users/Starlab/Desktop/psh/DATA/KITTI-360/data_3d_raw"
semantic_dir = "C:/Users/Starlab/Desktop/psh/DATA/KITTI-360/data_3d_semantics"
bbox_dir = "C:/Users/Starlab/Desktop/psh/DATA/KITTI-360/data_3d_bboxes"
output_dir = "C:/Users/Starlab/Desktop/psh/DATA/KITTI-360/preprocessed_data"


preprocess_data(raw_dir, semantic_dir, bbox_dir, output_dir, voxel_size=0.05)
