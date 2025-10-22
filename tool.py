import matplotlib.pyplot as plt
import pandas as pd

# 构造数据（从 LaTeX 表格中提取）
data = {
    "Models": [
        "Transformer", "BiLSTM", "TCN", "BiGRU",
        "CNN", "PointNet++", "MLP", "Shuffled Seq"
    ],
    "Accuracy": [0.9093, 0.9192, 0.8957, 0.9099, 0.8893, 0.8136, 0.8615, 0.4033],
    "F1": [0.8948, 0.9039, 0.8754, 0.8965, 0.8692, 0.7849, 0.8341, 0.3612],
    "Precision": [0.8835, 0.8964, 0.8677, 0.8849, 0.8619, 0.8065, 0.8342, 0.3901],
    "Recall": [0.9164, 0.9143, 0.8883, 0.9164, 0.8926, 0.8209, 0.8609, 0.4457],
}

# 创建 DataFrame
df = pd.DataFrame(data)
df.set_index("Models", inplace=True)
colors = ['#7B95C6', '#A2C986', '#F58020', '#C85E62']
# 设置图像属性
fig, ax = plt.subplots(figsize=(12, 4))  # 更紧凑尺寸

# 绘图
df.plot(kind='bar', ax=ax, width=0.65, color=colors)

# 美化样式
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Performance Comparison of Models", fontsize=11)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)
ax.legend(title="Metrics", loc='upper right', fontsize=9, title_fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.xticks(rotation=30, ha='right')
plt.tight_layout(pad=0.5)

# 保存图像（高分辨）
plt.savefig("temporal_model_comparison.png", dpi=300)
