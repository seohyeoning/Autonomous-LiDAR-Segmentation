import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split preprocessed files into train, val, and test datasets.
    :param data_dir: Path to the preprocessed data directory.
    :param train_ratio: Proportion of training data.
    :param val_ratio: Proportion of validation data.
    :return: Dictionary with train, val, and test splits.
    """
    # 모든 파일 가져오기 및 정렬
    all_files = sorted([file for file in os.listdir(data_dir) if file.endswith(".npz")])

    # 파일 개수 계산
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    # 데이터 나누기
    splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:]
    }

    return splits

class Kitti360Dataset(Dataset):
    def __init__(self, data_dir, file_list):
        """
        Initialize dataset for Kitti360 preprocessed data.
        :param data_dir: Path to the preprocessed data directory.
        :param file_list: List of .npz files to include in the dataset.
        """
        self.data_dir = data_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        return {
            "voxels": data["voxels"],
            "labels": data["labels"],
            "bounding_boxes": data["bounding_boxes"]
        }

def create_dataloader(data_dir, file_list, batch_size=4, shuffle=True, num_workers=4):
    """
    Create DataLoader for the given split.
    :param data_dir: Path to the preprocessed data directory.
    :param file_list: List of .npz files for the split.
    :param batch_size: Batch size for DataLoader.
    :param shuffle: Whether to shuffle the data.
    :param num_workers: Number of worker processes for DataLoader.
    :return: DataLoader object.
    """
    dataset = Kitti360Dataset(data_dir, file_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
