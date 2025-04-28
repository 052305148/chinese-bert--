import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pyvis.network import Network
from scipy.spatial.distance import squareform

# ========== 数据准备 ==========
# 读取词频数据（假设CSV格式为"词语,频率"）
df = pd.read_csv('word_freq.csv', header=None, names=['word', 'freq'])
words = df['word'].tolist()
freq_dict = df.set_index('word')['freq'].to_dict()

# ========== 生成模拟共现矩阵 ==========
n = len(words)
np.random.seed(42)  # 固定随机种子保证可重复性
co_occur = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(i, n):
        if i != j:
            # 基于词频生成模拟共现次数
            base = min(freq_dict[words[i]], freq_dict[words[j]])
            co_occur[i][j] = co_occur[j][i] = int(base * 0.02) + np.random.randint(0, 20)


# ========== 构建网络图 ==========
G = nx.Graph()
for word in words:
    G.add_node(word, size=float(freq_dict[word]) / 100)  # 确保节点大小是 float 类型



# 添加带权重的边
for i in range(n):
    for j in range(i+1, n):
        if co_occur[i][j] > 0:
            G.add_edge(words[i], words[j], weight=co_occur[i][j])

# ========== 核心-边缘分析 ==========
# 方法1：基于k-core分解
kcore = nx.k_core(G)
core_nodes_k = list(kcore.nodes())

# 方法2：基于度中心性
degree_centrality = nx.degree_centrality(G)
threshold = np.percentile(list(degree_centrality.values()), 75)  # 取前25%作为核心
core_nodes = [n for n, dc in degree_centrality.items() if dc >= threshold]

# ========== 结构对等聚类 ==========
# 将共现矩阵转换为距离矩阵（假设较大的共现次数对应较小的距离）
distance_matrix = np.max(co_occur) - co_occur
np.fill_diagonal(distance_matrix, 0)  # 确保对角线元素为零
X = squareform(distance_matrix)  # 压缩为一维格式
Z = linkage(X, 'ward')

# 自动确定聚类数（根据最大距离变化）
last_10 = Z[-10:, 2]
differences = np.diff(last_10)
n_clusters = differences.argmax() + 2

clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = clustering.fit_predict(distance_matrix)

# ========== 可视化 ==========
# 网络可视化（PyVis）
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

# 设置节点属性
for node in G.nodes():
    net.add_node(node,
                 size=float(G.nodes[node]['size']),  # 转换为 float
                 group=int(clusters[words.index(node)]),  # 确保是 int 类型
                 title=f"{node}<br>频率：{freq_dict[node]}")

# 添加边
for edge in G.edges(data=True):
    net.add_edge(edge[0], edge[1], value=edge[2]['weight']/10)

net.show_buttons(filter_=['physics'])
net.show("network.html")

# 树状图可视化
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=words, orientation='right', leaf_font_size=10)
plt.title('层次聚类树状图')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300)

# ========== 结果输出 ==========
print("核心节点（k-core）:", core_nodes_k)
print("核心节点（度中心性）:", core_nodes)
print("\n结构对等聚类结果：")
for i in range(n_clusters):
    cluster_words = [words[j] for j in range(n) if clusters[j] == i]
    print(f"聚类{i+1}: {cluster_words}")

# 保存分析结果
with open('analysis_results.txt', 'w') as f:
    f.write("核心节点（k-core）:\n" + "\n".join(core_nodes_k) + "\n\n")
    f.write("核心节点（度中心性）:\n" + "\n".join(core_nodes) + "\n\n")
    f.write("结构对等聚类:\n")
    for i in range(n_clusters):
        cluster_words = [words[j] for j in range(n) if clusters[j] == i]
        f.write(f"Cluster {i+1}: {', '.join(cluster_words)}\n")