import matplotlib.pyplot as plt
from plot import plot_logo

ax = plt.subplot(1, 1, 1)  # 5 行 1 列，选择第 1 个位置
plot_logo([1,2,3,4,5], ax=ax)  # 绘制特征重要性 logo 图
ax.set_title('Midpoint of the entire sequence:index=25', fontsize=14)
ax.tick_params(axis='both', labelsize=12)  # 设置坐标轴字体大小