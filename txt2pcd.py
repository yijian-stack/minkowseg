import open3d as o3d
import numpy as np
import os

def label_to_color(label):
    # 这里简单定义一些颜色，可以根据需要扩展
    label_color_map = {
        -1: [255, 255, 255],  # 未标记/背景为白色
        0: [255, 0, 0],      # 标签0为红色
        1: [0, 255, 0],      # 标签1为绿色
        16: [0, 0, 255]      # 标签16为蓝色
    }
    return label_color_map.get(label, [255, 255, 255])  # 默认为白色

def txt_to_pcd(input_file, output_file):
    points = []
    colors = []
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            # 首先将前三个坐标转换为float，然后读取第四个值作为label
            x, y, z = map(float, parts[:3])
            label = int(parts[3])  # 读取label，转换为整数
            points.append([x, y, z])
            # 转换标签为颜色
            colors.append(label_to_color(label))
    
    points = np.array(points)
    colors = np.array(colors) / 255.0  # 将颜色从0-255规范化到0-1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色

    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved {len(points)} points with colors to '{output_file}'")

def convert_all_txt_to_pcd(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace('.txt', '.pcd'))
            txt_to_pcd(input_file, output_file)

input_dir = 'data/predata'
output_dir = 'data/predata/pcd'
convert_all_txt_to_pcd(input_dir, output_dir)
