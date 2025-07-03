import matplotlib.pyplot as plt

gowalla_lightgcn_ndcg20 = []
with open('gowalla_LightGCN_seed2020_lr0.001_reg0.0001_dim64_nl2.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            gowalla_lightgcn_ndcg20.append(float(parts[7]))
print("gowalla_lightgcn_ndcg20:", gowalla_lightgcn_ndcg20)

gowalla_lightgcnpp_ndcg20_best = []
with open('gowalla_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha0.6_beta-0.1_gamma0.2.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            gowalla_lightgcnpp_ndcg20_best.append(float(parts[7]))
print("gowalla_lightgcnpp_ndcg20_best:", gowalla_lightgcnpp_ndcg20_best)

gowalla_lightgcnpp_ndcg20 = []
with open('gowalla_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha1.0_beta1.0_gamma0.5.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            gowalla_lightgcnpp_ndcg20.append(float(parts[7]))
print("gowalla_lightgcnpp_ndcg20:", gowalla_lightgcnpp_ndcg20)


epochs = list(range(1, len(gowalla_lightgcn_ndcg20) + 1))


plt.plot(epochs, gowalla_lightgcn_ndcg20, label='LightGCN NDCG@20')
plt.plot(epochs, gowalla_lightgcnpp_ndcg20_best, label='LightGCN++ NDCG@20 (Best)')
plt.plot(epochs, gowalla_lightgcnpp_ndcg20, label='LightGCN++ NDCG@20')


plt.xlabel('Epochs')
plt.ylabel('NDCG@20')
plt.title('gowalla')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()