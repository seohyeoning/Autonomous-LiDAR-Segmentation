# 기본 모듈
import os
import numpy as np
import argparse

# PyTorch 관련 모듈
import torch

# 데이터셋 및 데이터 로더 정의
from dataloader import split_dataset, create_dataloader  # dataset.py에서 정의한 클래스 및 함수
from visualization import visualize_results
from model import SECONDModel



def main(args):
   
    data_dir = "C:/Users/Starlab/Desktop/psh/DATA/KITTI-360/preprocessed_data"

    cfg_file = "C:/Users/Starlab/Desktop/psh/OpenPCDet/CONFIG/second.yaml"
    ckpt_file = "C:/Users/Starlab/Desktop/psh/OpenPCDet/CKPT/second_kitti.pth"
    output_dir = "C:/Users/Starlab/Desktop/psh/RESULT"

    # 데이터셋 나누기
    splits = split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1)
    
    # 데이터 로더 생성
    train_loader = create_dataloader(data_dir, split='train', batch_size=4)
    val_loader = create_dataloader(data_dir, split='val', batch_size=4)

    # 모델 초기화
    model = SECONDModel(cfg_file, ckpt_file)

    # 학습
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
    model.train(train_loader, optimizer, epochs=10)

    # 추론 및 결과 시각화
    for batch in val_loader:
        predictions = model.predict(batch)
        points = batch["voxels"].reshape(-1, 3)
        segmentation_labels = predictions[0]["segmentation_labels"]
        instance_labels = predictions[0]["instance_labels"]
        bounding_boxes = predictions[0]["bounding_boxes"]

        visualize_results(points, segmentation_labels, instance_labels, bounding_boxes)
        break  # 한 번만 시각화



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR Semantic and Instance Segmentation with SECOND")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed data directory")
    parser.add_argument("--cfg_file", type=str, required=True, help="Path to SECOND model config file")
    parser.add_argument("--ckpt_file", type=str, required=True, help="Path to pretrained SECOND model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    args = parser.parse_args()
    
    main(args)

