import os
import jieba
from collections import Counter
import re
from gensim.models import Word2Vec
import networkx as nx
from scipy.sparse import lil_matrix, save_npz
import numpy as np
import gc  # Garbage collector

def get_txt_files(folder_path):
    """获取指定文件夹中所有 txt 文件的路径列表"""
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

def read_and_segment(file_path):
    """读取文件内容并使用 jieba 进行中文分词"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            words = jieba.cut(content)
            return list(words)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

def is_chinese_word(word):
    """判断词语是否为中文词语（不包含数字和标点）"""
    return bool(re.match(r'^[\u4e00-\u9fa5]+$', word))

def train_word2vec_model(word_lists, vector_size=100, window=5, min_count=3, workers=16):
    """训练 Word2Vec 模型"""
    model = Word2Vec(
        sentences=word_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=32,
        sg=1  # 使用 Skip-Gram 模型，更适合捕捉词语间的语义关系
    )
    return model

def main():
    # 指定文件夹路径和参数
    folder_path = r'D:\专精特新文本'
    min_word_freq = 3  # 最小词频阈值，与 Word2Vec 的 min_count 保持一致
    threshold = 10     # 共现次数阈值

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在。")
        return

    # 获取所有 txt 文件路径
    txt_files = get_txt_files(folder_path)
    print(f"共找到 {len(txt_files)} 个 txt 文件。")

    # **第一步：计算词频并过滤词汇**
    print("第一步：计算词频并过滤词汇")
    word_freq = Counter()
    for i, file in enumerate(txt_files):
        if i % 100 == 0:
            print(f"词频统计：已处理 {i} 个文件")
        words = read_and_segment(file)
        filtered_words = [word for word in words if is_chinese_word(word)]
        word_freq.update(filtered_words)

    # 过滤低频词
    filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_word_freq}
    vocabulary = list(filtered_word_freq.keys())
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    # 保存词频结果
    with open('word_freq1.csv', 'w', encoding='utf-8') as f:
        for word, freq in sorted(filtered_word_freq.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{word},{freq}\n")
    print(f"词频统计完成，共 {len(vocabulary)} 个词，结果已保存至 'word_freq1.csv'。")

    # 清理内存
    del word_freq
    gc.collect()

    # **第二步：收集分词后的文档并训练 Word2Vec 模型**
    print("第二步：收集分词后的文档并训练 Word2Vec 模型")
    all_word_lists = []
    for i, file in enumerate(txt_files):
        if i % 100 == 0:
            print(f"收集分词：已处理 {i} 个文件")
        words = read_and_segment(file)
        filtered_words = [word for word in words if word in word_to_index]
        if len(filtered_words) > 0:
            all_word_lists.append(filtered_words)

    # 训练 Word2Vec 模型
    model = train_word2vec_model(all_word_lists, min_count=min_word_freq)
    model.save("word2vec_model.model")
    print("Word2Vec 模型训练完成，已保存至 'word2vec_model.model'")

    # **第三步：使用 Word2Vec 模型找出相似词语**
    print("第三步：找出与关键词相似的词语")
    target_words = ["专业化", "精细化", "特色化", "新颖"]
    similar_words = set()
    for word in target_words:
        if word in model.wv:
            similar = [w for w, _ in model.wv.most_similar(word, topn=25)]
            similar_words.update(similar)
            print(f"与 '{word}' 相似的词语：{[w for w in similar]}")
        else:
            print(f"词语 '{word}' 不在模型词汇表中。")
    similar_words.update(target_words)  # 包含关键词本身
    similar_words = list(similar_words)
    print(f"共选取 {len(similar_words)} 个词语用于构建共现矩阵和网络图。")

    # **第四步：创建基于相似词的共现矩阵**
    print("第四步：创建稀疏共现矩阵")
    similar_word_to_index = {word: i for i, word in enumerate(similar_words)}
    vocab_size = len(similar_words)
    cooccurrence_matrix = lil_matrix((vocab_size, vocab_size), dtype=np.int32)

    for doc_words in all_word_lists:
        filtered_doc_words = [word for word in doc_words if word in similar_word_to_index]
        if len(filtered_doc_words) > 0:
            doc_indices = [similar_word_to_index[w] for w in set(filtered_doc_words)]
            for i, idx1 in enumerate(doc_indices):
                for idx2 in doc_indices[i + 1:]:
                    cooccurrence_matrix[idx1, idx2] += 1
                    cooccurrence_matrix[idx2, idx1] += 1

    # 保存共现矩阵
    save_npz('cooccurrence_matrix.npz', cooccurrence_matrix.tocsr())
    print("共现矩阵已保存至 'cooccurrence_matrix.npz'。")

    # **第五步：构建网络图**
    print("第五步：构建网络图")
    G = nx.Graph()
    G.add_nodes_from(similar_words)
    cx = cooccurrence_matrix.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j and v >= threshold:
            G.add_edge(similar_words[i], similar_words[j], weight=float(v))

    # 保存网络图
    nx.write_gexf(G, 'cooccurrence_network.gexf')
    print("网络图已保存至 'cooccurrence_network.gexf'。")

    # 清理内存
    del all_word_lists, cooccurrence_matrix
    gc.collect()

if __name__ == '__main__':
    main()
