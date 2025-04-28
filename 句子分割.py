import os
import re
import csv

# 定义文件夹路径和输出CSV文件路径
folder_path = r"D:\11\专精特新文本"
csv_file_path = r"D:\sentences.csv"

# 获取文件夹中所有TXT文件列表
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 存储所有句子的列表
sentences = []

# 匹配中文句号、感叹号、问号，但排除数字中的小数点
sentence_end_regex = re.compile(r'''
    (?<![\d.])      # 前面不是数字或小数点
    ([。！？])      # 匹配中文标点
    (?![.])       # 后面不是小数点或数字
''', re.VERBOSE)

for txt_file in txt_files:
    file_path = os.path.join(folder_path, txt_file)
    try:
        # 尝试用多种编码打开文件
        for encoding in ['utf-8', 'gbk', 'gb18030', 'big5']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    # 预处理：合并换行符和空格
                    text = re.sub(r'\s+', ' ', text)
                    # 分割句子（排除数字中的小数点）
                    sentences_in_file = sentence_end_regex.split(text)
                    # 提取有效句子（保留分隔符）
                    sentences_in_file = [s.strip() for s, p in zip(sentences_in_file[::2], sentences_in_file[1::2])]
                    sentences_in_file = [s + p for s, p in zip(sentences_in_file, sentences_in_file[1::2])]
                    sentences_in_file = [s.strip() for s in sentences_in_file if s.strip()]
                    sentences.extend(sentences_in_file)
                    break
            except UnicodeDecodeError:
                continue
    except Exception as e:
        print(f"处理文件 {txt_file} 时发生错误：{e}")

# 写入CSV文件（UTF-8-BOM编码避免Excel乱码）
try:
    with open(csv_file_path, 'w', encoding='utf-8-sig', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["完整句子"])
        writer.writerows([[sentence] for sentence in sentences])
    print(f"处理完成，共提取{len(sentences)}个句子，结果已保存到 {csv_file_path}")
except Exception as e:
    print(f"写入CSV文件时发生错误：{e}")