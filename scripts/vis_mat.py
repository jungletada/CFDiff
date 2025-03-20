import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm


def visualized_csv(csv_file, base_path, target_path):
    csv_path = os.path.join(base_path, csv_file)
    data = pd.read_csv(csv_path, encoding='utf-8-sig')
    x = data['x-coordinate']  # 0, 0.1
    y = data['y-coordinate']  # 0, 0.01

    for k in data.columns:
        if k in ['pressure', 'x-velocity', 'y-velocity', 'temperature', 'velocity', 'contour']:
            value = data[k]
            plt.figure() 
            contour = plt.tricontourf(x, y, value, cmap='binary')
            plt.gca().set_aspect('equal', 'box')
            plt.axis('off')  # Turn off axis
            plt.tight_layout()  # Adjust layout to remove extra whitespace
            save_path = os.path.join(target_path, k, csv_file.replace('.csv', '.tiff'))
            plt.savefig(save_path, dpi=800, transparent=True, format='tiff')
            plt.close()


def clip_and_resize(tiff_file, base_path, target_path):
    # Step 1: 读取 tiff 图片并剔除透明像素
    tiff_path = os.path.join(base_path, tiff_file)
    image = Image.open(tiff_path)
    image_array = np.array(image)
    alpha_channel = image_array[...,-1]

    # 找到不透明部分的四个顶点坐标
    non_transparent_pixels = np.argwhere(alpha_channel == 255)
    top_left = np.min(non_transparent_pixels, axis=0)
    bottom_right = np.max(non_transparent_pixels, axis=0)

    # 根据顶点坐标对 image_array 进行切片
    cropped_image_array = image_array[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, :]
    height, width, _ = cropped_image_array.shape
    left_crop_array = cropped_image_array[:, :width//2, 0]

    c_image = Image.fromarray(left_crop_array)
    c_image.save(os.path.join(target_path, tiff_file.replace('.tiff', 's.tiff')))


if __name__ == '__main__':
    case_dir = 'case_data1'
    source_folder = f'data/{case_dir}/fluent_data_csv'
    os.makedirs(f"data/{case_dir}/fluent_data_vis", exist_ok=True)

    for keyword in ['pressure', 'x-velocity', 'y-velocity', 'temperature', 'velocity', 'contour']:
        target_folder = f"data/{case_dir}/fluent_data_vis/{keyword}"
        os.makedirs(target_folder, exist_ok=True)
    
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        visualized_csv(csv_file, source_folder, target_path=f"data/{case_dir}/fluent_data_vis")
    
    for keyword in ['pressure', 'x-velocity', 'y-velocity', 'temperature', 'velocity', 'contour']:
        source_folder = f"data/{case_dir}/fluent_data_vis/{keyword}"
        target_folder = f"data/{case_dir}/fluent_data_fig/{keyword}"
        os.makedirs(target_folder, exist_ok=True)
        tiff_files = [f for f in os.listdir(source_folder) if f.endswith('.tiff')]
        for tiff_file in tqdm(tiff_files, desc="Processing TIFF Files", unit="file"):
            clip_and_resize(tiff_file, source_folder, target_folder)