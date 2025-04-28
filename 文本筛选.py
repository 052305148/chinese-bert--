import os
import shutil
import pandas as pd

# 定义文件路径
csv_file_path = '号码.csv'
source_folder_path = 'D:\\2019_3936份'
target_folder_path = 'D:\\专精特新文本'

# 读取CSV文件中的数据，并将特定列指定为字符串类型以保留前导零
numbers_df = pd.read_csv(csv_file_path, dtype=str)

# 打印 DataFrame 的列名以检查它们
print("CSV 文件的列名:", numbers_df.columns)

# 检查数据框结构（调试用）
print(numbers_df.head())

# 处理假设CSV文件中的列名为 'C1' 的情形
if 'C1' in numbers_df.columns:
    number_list = numbers_df['C1'].tolist()
else:
    print("列名 'C1' 不存在，请检查 CSV 文件的实际列名。")
    # 假设需要的列名指向第一个存在的列
    number_list = numbers_df.iloc[:, 0].tolist()

# 确保目标文件夹存在
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 添加调试信息
print(f"CSV 数字列表: {number_list}")

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    # 仅处理txt文件
    if filename.endswith('.txt'):
        # 截取文件名前六位
        file_number = filename[:6]
        # 添加调试信息
        print(f"正在检查文件: {filename}, 前六位: {file_number}")

        # 检查文件名前六位是否在CSV中存在
        if file_number in number_list:
            # 文件路径准备
            source_file_path = os.path.join(source_folder_path, filename)
            target_file_path = os.path.join(target_folder_path, filename)

            # 复制文件
            shutil.copy(source_file_path, target_file_path)
            print(f"已复制文件: {filename}")

print("任务完成")