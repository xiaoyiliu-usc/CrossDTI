# import pandas as pd
# import numpy as np
# from collections import Counter
# import os
#
#
# def iterative_train_test_split(X, y, train_size=0.7, val_size=0.15, random_state=None):
#     """
#     使用迭代分层采样策略为多标签数据创建数据集划分
#
#     参数:
#         X: 样本索引数组
#         y: 标签矩阵，形状为(n_samples, n_labels)
#         train_size: 训练集所占比例
#         val_size: 验证集所占比例
#         random_state: 随机种子
#
#     返回:
#         train_indices, val_indices, test_indices: 训练集、验证集和测试集的索引
#     """
#
#     if random_state is not None:
#         np.random.seed(random_state)
#
#     n_samples, n_labels = y.shape
#     test_size = 1 - train_size - val_size
#
#     # 创建三个数据集的空索引列表
#     train_indices = []
#     val_indices = []
#     test_indices = []
#
#     # 计算每个标签的期望分布
#     desired_train_dist = {j: train_size for j in range(n_labels)}
#     desired_val_dist = {j: val_size for j in range(n_labels)}
#     desired_test_dist = {j: test_size for j in range(n_labels)}
#
#     # 计算每个样本当前标签数量
#     sample_label_counts = np.sum(y, axis=1)
#
#     # 创建样本索引到标签的映射
#     sample_to_labels = {}
#     for i in range(n_samples):
#         sample_to_labels[i] = np.where(y[i] == 1)[0]
#
#     remaining_indices = set(range(n_samples))
#
#     # 首先处理标签组合频率最低的样本
#     # 这有助于保留稀有标签组合
#     sample_by_label_combination = {}
#     for i, labels in sample_to_labels.items():
#         labels_tuple = tuple(sorted(labels))
#         if labels_tuple not in sample_by_label_combination:
#             sample_by_label_combination[labels_tuple] = []
#         sample_by_label_combination[labels_tuple].append(i)
#
#     # 按组合频率排序
#     label_combinations_by_freq = sorted(
#         sample_by_label_combination.keys(),
#         key=lambda x: len(sample_by_label_combination[x])
#     )
#
#     # 迭代分配稀有标签组合
#     for combo in label_combinations_by_freq:
#         if len(sample_by_label_combination[combo]) < 5:  # 只处理频率低于5的组合
#             indices = sample_by_label_combination[combo]
#             np.random.shuffle(indices)
#
#             # 按照期望比例分配这些样本
#             n_train = max(1, int(train_size * len(indices)))
#             n_val = max(1, int(val_size * len(indices)))
#
#             train_indices.extend(indices[:n_train])
#             val_indices.extend(indices[n_train:n_train + n_val])
#             test_indices.extend(indices[n_train + n_val:])
#
#             # 从待处理集合中移除
#             for idx in indices:
#                 if idx in remaining_indices:
#                     remaining_indices.remove(idx)
#
#     # 计算当前每个类别的分布情况
#     def get_label_distribution(indices):
#         if not indices:
#             return {j: 0 for j in range(n_labels)}
#
#         dist = {}
#         for j in range(n_labels):
#             dist[j] = sum(y[i, j] for i in indices) / sum(y[:, j])
#         return dist
#
#     # 迭代处理剩余样本
#     while remaining_indices:
#         # 计算当前分布
#         train_dist = get_label_distribution(train_indices)
#         val_dist = get_label_distribution(val_indices)
#         test_dist = get_label_distribution(test_indices)
#
#         # 找出分布最不平衡的标签
#         label_diff_train = {j: desired_train_dist[j] - train_dist[j] for j in range(n_labels)}
#         label_diff_val = {j: desired_val_dist[j] - val_dist[j] for j in range(n_labels)}
#         label_diff_test = {j: desired_test_dist[j] - test_dist[j] for j in range(n_labels)}
#
#         # 选择最需要调整的标签
#         most_needed_train = max(label_diff_train.items(), key=lambda x: x[1])[0]
#         most_needed_val = max(label_diff_val.items(), key=lambda x: x[1])[0]
#         most_needed_test = max(label_diff_test.items(), key=lambda x: x[1])[0]
#
#         # 找出在剩余样本中包含这些标签的样本
#         candidates_train = [i for i in remaining_indices if y[i, most_needed_train] == 1]
#         candidates_val = [i for i in remaining_indices if y[i, most_needed_val] == 1]
#         candidates_test = [i for i in remaining_indices if y[i, most_needed_test] == 1]
#
#         # 如果没有找到候选样本，则随机选择
#         if not candidates_train and not candidates_val and not candidates_test:
#             index = np.random.choice(list(remaining_indices))
#             # 随机分配到一个数据集
#             r = np.random.random()
#             if r < train_size / (train_size + val_size + test_size):
#                 train_indices.append(index)
#             elif r < (train_size + val_size) / (train_size + val_size + test_size):
#                 val_indices.append(index)
#             else:
#                 test_indices.append(index)
#         else:
#             # 确定哪个集合更需要调整
#             max_diff_train = label_diff_train[most_needed_train]
#             max_diff_val = label_diff_val[most_needed_val]
#             max_diff_test = label_diff_test[most_needed_test]
#
#             if max_diff_train >= max_diff_val and max_diff_train >= max_diff_test and candidates_train:
#                 index = np.random.choice(candidates_train)
#                 train_indices.append(index)
#             elif max_diff_val >= max_diff_test and candidates_val:
#                 index = np.random.choice(candidates_val)
#                 val_indices.append(index)
#             elif candidates_test:
#                 index = np.random.choice(candidates_test)
#                 test_indices.append(index)
#             else:
#                 # 如果没有找到候选样本，则随机选择
#                 index = np.random.choice(list(remaining_indices))
#                 r = np.random.random()
#                 if r < train_size / (train_size + val_size + test_size):
#                     train_indices.append(index)
#                 elif r < (train_size + val_size) / (train_size + val_size + test_size):
#                     val_indices.append(index)
#                 else:
#                     test_indices.append(index)
#
#         # 从剩余索引中移除
#         remaining_indices.remove(index)
#
#     return train_indices, val_indices, test_indices
#
#
# def create_balanced_data_split(data_path='./data/processed_dataset3.csv',
#                                output_path='./data/balanced_data_index.txt',
#                                train_ratio=0.8,
#                                val_ratio=0.1,
#                                random_seed=42):
#     """
#     创建一个均衡的数据分割，并保存到文件
#
#     参数:
#         data_path: 数据集CSV的路径
#         output_path: 输出分割索引的文件路径
#         train_ratio: 训练集比例
#         val_ratio: 验证集比例
#         random_seed: 随机种子
#     """
#     print(f"正在从 {data_path} 加载数据...")
#     df = pd.read_csv(data_path)
#
#     # 解析标签
#     labels = df['label'].apply(lambda x: [int(i) for i in x.split(',')]).tolist()
#
#     # 创建标签矩阵
#     all_classes = set()
#     for label_list in labels:
#         all_classes.update(label_list)
#     num_classes = max(all_classes) + 1
#
#     # 创建标签矩阵
#     label_matrix = np.zeros((len(df), num_classes), dtype=int)
#     for i, label_list in enumerate(labels):
#         for label in label_list:
#             label_matrix[i, label] = 1
#
#     # 计算每个类别的分布
#     label_counts = np.sum(label_matrix, axis=0)
#     print("\n原始类别分布:")
#     for i in range(num_classes):
#         print(f"类别 {i}: {label_counts[i]} 样本 ({label_counts[i] / len(df) * 100:.1f}%)")
#
#     # 使用迭代分层采样分割数据
#     print("\n正在使用迭代分层采样分割数据...")
#     train_indices, val_indices, test_indices = iterative_train_test_split(
#         np.arange(len(df)),
#         label_matrix,
#         train_size=train_ratio,
#         val_size=val_ratio,
#         random_state=random_seed
#     )
#
#     # 分析分割结果
#     def analyze_split(indices, name):
#         if not indices:
#             print(f"{name} 为空!")
#             return
#
#         split_label_matrix = label_matrix[indices]
#         split_counts = np.sum(split_label_matrix, axis=0)
#
#         print(f"\n{name} 统计:")
#         print(f"样本数量: {len(indices)} ({len(indices) / len(df) * 100:.1f}%)")
#
#         for i in range(num_classes):
#             if label_counts[i] > 0:
#                 percent = split_counts[i] / label_counts[i] * 100
#                 print(f"类别 {i}: {split_counts[i]}/{label_counts[i]} ({percent:.1f}%)")
#
#     # 分析结果
#     analyze_split(train_indices, "训练集")
#     analyze_split(val_indices, "验证集")
#     analyze_split(test_indices, "测试集")
#
#     # 检查完整性
#     assert len(train_indices) + len(val_indices) + len(test_indices) == len(df), "分割不完整"
#     assert len(set(train_indices) & set(val_indices)) == 0, "训练集和验证集存在重叠"
#     assert len(set(train_indices) & set(test_indices)) == 0, "训练集和测试集存在重叠"
#     assert len(set(val_indices) & set(test_indices)) == 0, "验证集和测试集存在重叠"
#
#     # 保存分割结果
#     print(f"\n保存分割结果到 {output_path}...")
#
#     # 排序索引以保持一致性
#     train_indices = sorted(train_indices)
#     val_indices = sorted(val_indices)
#     test_indices = sorted(test_indices)
#
#     data_split = [train_indices, val_indices, test_indices]
#
#     with open(output_path, 'w') as f:
#         f.write(str(data_split))
#
#     print("完成!")
#     return data_split
#
#
# if __name__ == "__main__":
#     create_balanced_data_split()
import os

#
# def convert_index_format():
#     """
#     将balanced_data_index.txt文件中的嵌套列表转换为与原始索引格式相同的格式
#     """
#     # 读取balanced_data_index.txt
#     with open('./data/balanced_data_index.txt', 'r') as f:
#         content = f.read()
#
#     # 解析嵌套列表结构
#     # 注意: 这种方法在处理大型列表时可能不是最佳选择，但对于当前数据是有效的
#     splits = eval(content)
#
#     # 确保它是一个包含三个列表的列表
#     if len(splits) != 3:
#         raise ValueError("均衡数据索引应该包含三个列表 (训练集, 验证集, 测试集)")
#
#     # 创建新文件
#     with open('./data/balanced_data_index3.txt', 'w') as f:
#         # 分别写入三个列表，每个列表一行
#         for split in splits:
#             f.write(str(split) + '\n')
#
#     print(f"已成功将均衡数据索引转换为原始格式，并保存至 './data/balanced_data_index3.txt'")
#
# if __name__ == "__main__":
#     convert_index_format()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 直接使用提供的数据
data = {
    'MVP-DDI (PhysChem)': [72.3, 74.9, 74.3, 73.8],
    'MVP-DDI (Graph)': [80.0, 82.8, 81.4, 81.3],
    'MVP-DDI (SMILES)': [84.6, 87.2, 88.0, 86.6],
    'MVP-DDI (FP)': [89.9, 92.1, 93.2, 92.1],
    'MVP-DDI (Concatenate)': [90.8, 92.9, 93.3, 92.3],
    'MVP-DDI': [90.9, 93.1, 93.4, 92.4]
}

# 创建DataFrame
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
pivot_df = pd.DataFrame(data, index=metrics)

# 设置图表样式
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 18})

# 设置模型变体的颜色 - 为每个模型指定不同的颜色（按新的顺序）
model_colors = {
    'MVP-DDI (PhysChem)': '#f9e9d9',  # 红色
    'MVP-DDI (Graph)': '#b08a76',  # 棕色
    'MVP-DDI (SMILES)': '#9fc5d8',  # 蓝色
    'MVP-DDI (FP)': '#e3edf9',  # 绿色
    'MVP-DDI (Concatenate)': '#4f4567',  # 紫色
    'MVP-DDI': '#e6703e'  # 橙色
}

# 创建分组柱状图
bar_width = 0.15
x = np.arange(len(pivot_df.index))

# 画柱状图 - 为每个模型变体绘制一组柱子（按照列的顺序）
for i, model in enumerate(pivot_df.columns):
    plt.bar(x + i * bar_width, pivot_df[model], width=bar_width, color=model_colors[model],
            label=model, edgecolor='black', linewidth=0.5)

    # 在柱状图顶部添加数值标注
    for j, (metric, value) in enumerate(pivot_df[model].items()):
        text_y = value + 0.5  # 在柱子之上
        plt.text(x[j] + i * bar_width, text_y, f'{value:.1f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加标签和标题
plt.xlabel('Evaluation Metrics', fontsize=20, fontweight='bold')
plt.ylabel('Performance (%)', fontsize=20, fontweight='bold')

# 调整标题位置，增加上方的空间防止被图例挡住
plt.title('Ablation Study Comparison', fontsize=22, fontweight='bold', pad=50)

# 设置x轴标签
plt.xticks(x + bar_width * (len(pivot_df.columns) - 1) / 2, pivot_df.index, rotation=0)

# 调整y轴范围，为数值标注预留空间
plt.ylim(70, 98)

# 设置y轴网格线，增强可读性
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 将图例移到顶部，确保不会覆盖标题，按照顺序排列
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True)

# 调整布局，确保所有元素都显示完整
plt.tight_layout()

# 增加顶部边距，为标题和图例提供更多空间
plt.subplots_adjust(top=0.85)

# 保存图表
plt.savefig('ablation_comparison.png', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from collections import Counter
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 疾病标签映射
# label_mapping = {
#     9: "L9",
#     8: "L8",
#     10: "L10",
#     2: "L2",
#     4: "L4",
#     7: "L7",
#     0: "L0",
#     3: "L3",
#     1: "L1",
#     5: "L5",
#     6: "L6"
# }
#
# # 加载数据集
# df = pd.read_csv("./data/dataset.csv")
# print(f"数据集形状: {df.shape}")
# print(f"前几行数据:\n{df.head()}")
#
#
# # 解析标签函数 - 支持多种可能的格式
# def parse_labels(label_str):
#     """将标签字符串解析为整数列表"""
#     try:
#         # 如格式为 "[0, 3, 4]"
#         if isinstance(label_str, str) and label_str.startswith('[') and label_str.endswith(']'):
#             return [int(x.strip()) for x in label_str[1:-1].split(',') if x.strip()]
#         # 如格式为 "0,3,4"
#         elif isinstance(label_str, str) and ',' in label_str:
#             return [int(x.strip()) for x in label_str.split(',') if x.strip()]
#         # 如格式为单个数字字符串 "3"
#         elif isinstance(label_str, str) and label_str.strip().isdigit():
#             return [int(label_str.strip())]
#         # 如已经是数字
#         elif isinstance(label_str, (int, float)) and not np.isnan(label_str):
#             return [int(label_str)]
#         else:
#             return []
#     except:
#         print(f"解析标签出错: {label_str}")
#         return []
#
#
# # 解析所有标签并计数
# all_labels = []
# for label in df['label']:
#     parsed_labels = parse_labels(label)
#     all_labels.extend(parsed_labels)
#
# label_counts = Counter(all_labels)
#
# # 创建用于绘图的数据框
# counts_df = pd.DataFrame({
#     'disease': [label_mapping.get(label, f"未知 ({label})") for label in label_counts.keys()],
#     'count': label_counts.values()
# }).sort_values('count', ascending=False)
#
# # 输出每种疾病的数量
# print("\n疾病分布:")
# for disease, count in zip(counts_df['disease'], counts_df['count']):
#     print(f"{disease}: {count} 个样本")
#
# # 可视化1: 疾病标签计数柱状图
# plt.figure(figsize=(14, 8))
# ax = sns.barplot(x='disease', y='count', data=counts_df)
#
# # 在每个柱子顶部添加数值标签
# for i, p in enumerate(ax.patches):
#     height = p.get_height()
#     ax.text(
#         p.get_x() + p.get_width() / 2.,  # 柱子中心位置的x坐标
#         height + 0.2,                     # 柱子顶部往上一点的y坐标
#         f'{int(height)}',                 # 显示的文本（数值）
#         ha="center",                      # 水平对齐方式
#         fontsize=18,                      # 字体大小
#         fontweight='bold'                 # 字体加粗
#     )
#
# plt.title('疾病标签分布柱状图', fontsize=26)
# plt.xlabel('疾病标签', fontsize=24)
# plt.ylabel('数量', fontsize=24)
# plt.xticks(rotation=45, ha='right', fontsize=18)  # 增加了横轴标签字体大小
# plt.tick_params(axis='x', labelsize=18)  # 确保x轴刻度标签字体增大
# plt.tick_params(axis='y', labelsize=16)  # 也适当增加y轴刻度标签字体
# plt.tight_layout()
# plt.savefig('disease_distribution_bar.png')
# plt.show()
#
# # 可视化2: 百分比分布饼图
# plt.figure(figsize=(12, 12))
#
# # 使用 seaborn 的 pastel 配色方案 - 柔和专业的色调
# colors = sns.color_palette('pastel')
#
# plt.pie(
#     counts_df['count'],
#     labels=counts_df['disease'],
#     autopct='%1.1f%%',
#     colors=colors,
#     startangle=140,
#     textprops={'fontsize': 12}
# )
#
# plt.title('疾病标签分布饼图', fontsize=26)
# plt.tight_layout()
# plt.savefig('disease_distribution_pie.png')
# plt.show()
#
# print("\n可视化完成! 请检查保存的图像文件。")

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib as mpl
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
# # 数据准备 - 按照要求的顺序：GAT, AutoMSR, GCN, MVP-DDI
# models = ['GAT', 'AutoMSR', 'GCN', 'MVP-DDI(本模型)']
# metrics = ['准确率', '精确率', '召回率', 'F1分数']
#
# # 每个模型的四个指标数据
# gat_data = [40.5, 53.3, 40.5, 35.0]
# automsr_data = [63.3, 65.2, 65.1, 64.4]
# gcn_data = [96.0, 85.0, 75.5, 80.1]
# mvp_ddi_data = [90.7, 92.6, 93.4, 92.2]
#
# # 合并数据以便于排序 - 使用有序字典保持指定顺序
# from collections import OrderedDict
#
# model_data_dict = OrderedDict([
#     ('GAT', gat_data),
#     ('AutoMSR', automsr_data),
#     ('GCN', gcn_data),
#     ('MVP-DDI(本模型)', mvp_ddi_data)
# ])
#
# # 设置图形尺寸
# plt.figure(figsize=(12, 8))
#
# # 使用更加美观的配色方案（保证MVP-DDI颜色最深最明显）
# colors = {
#     'MVP-DDI(本模型)': '#4285F4',  # Google蓝色
#     'GCN': '#34A853',  # 绿色
#     'AutoMSR': '#FF8A65',  # 红色
#     'GAT': '#FBBC05'  # 黄色
# }
#
# # 设置柱状图的位置
# x = np.arange(len(metrics))
# width = 0.2  # 柱子宽度
#
# # 对于每个指标，绘制按指定顺序的模型
# for i, metric in enumerate(metrics):
#     # 获取当前指标下各模型的性能值
#     metric_values = {model: data[i] for model, data in model_data_dict.items()}
#
#     # 按指定顺序绘制柱状图（GAT, AutoMSR, GCN, MVP-DDI）
#     sorted_models = list(model_data_dict.keys())
#
#     # 绘制按指定顺序的柱状图
#     for j, model in enumerate(sorted_models):
#         bar_position = x[i] + width * (j - 1.5)
#         bar_height = metric_values[model]
#
#         plt.bar(bar_position, bar_height, width,
#                 label=model if i == 0 else "",
#                 color=colors[model])
#
#         # 在柱状图顶部添加数值标注
#         plt.text(bar_position, bar_height + 1, f'{bar_height}%',
#                  ha='center', va='bottom', fontsize=10, fontweight='bold')
#
# # 添加标题和标签
# plt.title('不同模型在评价指标上的性能对比', fontsize=22, fontweight='bold')
# plt.xlabel('评价指标', fontsize=20)
# plt.ylabel('百分比 (%)', fontsize=20)
# plt.xticks(x, metrics, fontsize=16)
# plt.yticks(fontsize=16)
#
# # 添加网格线使图表更易读
# plt.grid(axis='y', linestyle='--', alpha=0.7)
#
# # 设置y轴范围（增加上限以容纳数值标注）
# plt.ylim(0, 110)
#
# # 调整布局，增加底部边距以容纳图例
# plt.subplots_adjust(bottom=0.2)
#
# # 添加图例（只显示一次）
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
#
# # 保持图例顺序与模型顺序一致
# ordered_handles = []
# ordered_labels = []
# for model in models:
#     if model in by_label:
#         ordered_handles.append(by_label[model])
#         ordered_labels.append(model)
#
# plt.legend(ordered_handles, ordered_labels, loc='upper center',
#            bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)
#
# plt.show()