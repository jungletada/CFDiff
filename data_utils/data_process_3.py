import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

left = 640
top = 43
DPI = 580

CMAP = {'pressure': 'viridis',
        'x-velocity': 'plasma', 
        'y-velocity': 'winter',
        'velocity': 'GnBu',
        'temperature': 'coolwarm',
        }
case_dir = 'case_data1'
source_folder = f'data/{case_dir}/fluent_data_csv'
target_folder = f'fluent_data_map'

# Parse min and max values from stat.txt
min_vals = {'pressure':-37.73662186,
            'x-velocity':-0.2224825025, 
            'y-velocity':-0.2417525947,
            'velocity': 0.0,
            'temperature':299.9764404}

max_vals = {'pressure':57.6361618,
            'x-velocity':0.3928118348, 
            'y-velocity':0.2281097323,
            'velocity': 0.3930110071636349,
            'temperature':310.3595276}

# Columns to normalize
cols = list(max_vals.keys())
cols.append('contour')

def csv_to_image(csv_file):
    # Normalize and visualize each column
    file_path = os.path.join(source_folder, csv_file)
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    for col in cols:
        save_path = os.path.join(target_folder, col)
        save_img_path = os.path.join(save_path, csv_file.replace('csv', 'png'))
        # if os.path.exists(save_img_path):
        #     continue
        # Min-max normalization
        if col != 'contour':
            norm = (df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
            plt.figure()
            plt.axis('off')
            plt.tight_layout()
            plt.gca().set_aspect('equal', 'box')
            plt.tricontourf(
                df['x-coordinate'], df['y-coordinate'], norm, cmap=CMAP[col])
            plt.savefig(save_img_path, 
                        bbox_inches='tight', 
                        pad_inches=-0.01, 
                        dpi=DPI)
            plt.clf()
            plt.close()
            img = Image.open(save_img_path)
            cropped_img = img.crop((left, top, left+512, top+256))
            cropped_img.save(save_img_path)
            
        else:
            plt.figure()
            plt.axis('off')
            plt.tight_layout()
            plt.gca().set_aspect('equal', 'box')
            plt.tricontourf(
                df['x-coordinate'], df['y-coordinate'], df['contour'], cmap='gray')
            plt.savefig(save_img_path, 
                        bbox_inches='tight', 
                        pad_inches=-0.01, 
                        dpi=DPI)
            plt.clf()
            plt.close()
            img = Image.open(save_img_path)
            cropped_img = img.crop((left, top, left+512, top+256))
            cropped_img.save(save_img_path)
        

if __name__ == '__main__':
    os.makedirs(target_folder, exist_ok=True)
    for col in cols:
        os.makedirs(os.path.join(target_folder, col), exist_ok=True)
    csv_to_image(csv_file='190.csv')
    # csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    # for csv_file in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
    #     csv_to_image(csv_file)
