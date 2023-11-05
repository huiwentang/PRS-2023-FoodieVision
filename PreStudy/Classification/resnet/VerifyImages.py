import os
from PIL import Image

# 指定数据集文件夹路径和图像文件扩展名
dataset_folder = 'D:/Software/data_set/food_data2/train'  # 替换为你的数据集文件夹路径
image_extension = '.jpg'  # 替换为你的图像文件扩展名

# 获取数据集文件夹中的所有图像文件路径
image_paths = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder) if filename.endswith(image_extension)]

# 遍历每个图像文件并尝试打开
for image_path in image_paths:
    try:
        # 尝试打开图像文件
        img = Image.open(image_path)
        # 如果成功打开图像，可以在这里添加任何你需要执行的操作
        # 例如，可以将图像添加到一个列表或记录日志
    except Exception as e:
        # 如果打开图像时出现错误，记录错误信息和文件路径
        print(f"Error opening image: {image_path}")
        print(f"Error message: {str(e)}")
