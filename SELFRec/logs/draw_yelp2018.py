import matplotlib.pyplot as plt

# 初始化空列表存储LightGCN模型的NDCG@20数据
yelp2018_lightgcn_ndcg20 = []
# 读取LightGCN模型数据文件
with open('yelp2018_LightGCN_seed2020_lr0.001_reg0.0001_dim64_nl2.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        # 筛选test行并提取第8列（索引7）的NDCG@20数据
        if len(parts) >= 8 and parts[1] == 'test':
            yelp2018_lightgcn_ndcg20.append(float(parts[7]))
print("yelp2018_lightgcn_ndcg20:", yelp2018_lightgcn_ndcg20)

# 初始化空列表存储LightGCN++（Best）模型的NDCG@20数据
yelp2018_lightgcnpp_ndcg20_best = []
# 读取LightGCN++（Best）模型数据文件
with open('yelp2018_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha0.6_beta-0.1_gamma0.0.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            yelp2018_lightgcnpp_ndcg20_best.append(float(parts[7]))
print("yelp2018_lightgcnpp_ndcg20_best:", yelp2018_lightgcnpp_ndcg20_best)

# 初始化空列表存储LightGCN++模型的NDCG@20数据
yelp2018_lightgcnpp_ndcg20 = []
# 读取LightGCN++模型数据文件
with open('yelp2018_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha1.0_beta1.0_gamma0.5.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            yelp2018_lightgcnpp_ndcg20.append(float(parts[7]))
print("yelp2018_lightgcnpp_ndcg20:", yelp2018_lightgcnpp_ndcg20)

# 生成epochs列表（从1开始，与数据长度一致）
epochs = list(range(1, len(yelp2018_lightgcn_ndcg20) + 1))

# 绘制三条曲线
plt.plot(epochs, yelp2018_lightgcn_ndcg20, label='LightGCN NDCG@20')
plt.plot(epochs, yelp2018_lightgcnpp_ndcg20_best, label='LightGCN++ NDCG@20 (Best)')
plt.plot(epochs, yelp2018_lightgcnpp_ndcg20, label='LightGCN++ NDCG@20')

# 设置图表属性
plt.xlabel('Epochs')
plt.ylabel('NDCG@20')
plt.title('yelp2018')  # 标题为Yelp2018数据集
plt.xticks(epochs)  # 横坐标显示为整数epochs
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线

# 显示图像
plt.show()