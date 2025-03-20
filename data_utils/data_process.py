import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def add_velocity_and_contour(input_csv, output_csv, eps=1e-12):
    """
    在 input_csv 中添加：
    1. velocity 列: 速度模长 sqrt(x-velocity^2 + y-velocity^2)
    2. contour 列: velocity == 0 且 y 不等于 (min_y, max_y) => 1, 否则 0

    :param input_csv:  原始 CSV 文件路径，如 '101.csv'
    :param output_csv: 生成的新 CSV 文件路径，如 '101_with_vel_contour.csv'
    :param eps:        判断 y 是否接近最小/最大值时使用的阈值
    """
    # 1. 读取 CSV
    data = pd.read_csv(input_csv, encoding='utf-8-sig')
    # 3. 如果列名前后有空格，可以做 strip
    data.columns = [col.strip() for col in data.columns]
    # print("去掉空格后列名:", data.columns.tolist())
    # 2. 计算 velocity
    value_x = data['x-velocity']
    value_y = data['y-velocity']
    data['velocity'] = np.sqrt(value_x**2 + value_y**2)  # 保存在 'velocity' 列中
    
    # 3. 找到 y 的最小值和最大值
    y_min = data['y-coordinate'].min()
    y_max = data['y-coordinate'].max()
    
    # 4. 定义判断函数，用于比较浮点数是否接近 y_min 或 y_max
    def is_near(a, b, tol=eps):
        return abs(a - b) <= tol
    
    # 5. 生成 contour 列
    # 条件：
    #  1) velocity == 0
    #  2) y 不接近 y_min 且 不接近 y_max
    # 则 contour = 1, 否则 0
    contour_list = []
    for idx, row in data.iterrows():
        vel = row['velocity']
        y_val = row['y-coordinate']
        
        if vel == 0 and not is_near(y_val, y_min) and not is_near(y_val, y_max):
            contour_list.append(0)
        else:
            contour_list.append(1)
    
    data['contour'] = contour_list
    
    # 6. 写回 CSV
    data.to_csv(output_csv, index=False, encoding='utf-8-sig')
    # print(f"[Done] Add 'velocity' and 'contour', and save to: {output_csv}")


if __name__ == '__main__':
    case_dir = 'case_data2'
    source_folder = f'data/{case_dir}/fluent_data_csv'
    target_folder = f"data/{case_dir}/fluent_data_all_csv"

    os.makedirs(target_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    for csv_file in tqdm(csv_files, desc="Processing CSV Files", unit="file"):
        input_file = os.path.join(source_folder, csv_file)
        output_file = os.path.join(target_folder, csv_file)
        add_velocity_and_contour(input_file, output_file)