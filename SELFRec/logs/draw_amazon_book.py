import matplotlib.pyplot as plt

amazon_book_lightgcn_ndcg20 = []
with open('amazon-book_LightGCN_seed2020_lr0.001_reg0.0001_dim64_nl2.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            amazon_book_lightgcn_ndcg20.append(float(parts[7]))
print("amazon_book_lightgcn_ndcg20:", amazon_book_lightgcn_ndcg20)


amazon_book_lightgcnpp_ndcg20_best = []
with open('amazon-book_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha0.6_beta-0.1_gamma0.2copy.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            amazon_book_lightgcnpp_ndcg20_best.append(float(parts[7]))
print("amazon_book_lightgcnpp_ndcg20_best:", amazon_book_lightgcnpp_ndcg20_best)


amazon_book_lightgcnpp_ndcg20 = []
with open('amazon-book_LightGCNpp_seed2020_lr0.001_reg0.0001_dim64_nl2_alpha1.0_beta1.0_gamma0.5.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) >= 8 and parts[1] == 'test':
            amazon_book_lightgcnpp_ndcg20.append(float(parts[7]))
print("amazon_book_lightgcnpp_ndcg20:", amazon_book_lightgcnpp_ndcg20)


epochs = list(range(1, len(amazon_book_lightgcn_ndcg20) + 1))

plt.plot(epochs, amazon_book_lightgcn_ndcg20, label='LightGCN NDCG@20')
plt.plot(epochs, amazon_book_lightgcnpp_ndcg20_best, label='LightGCN++ NDCG@20 (Best)')
plt.plot(epochs, amazon_book_lightgcnpp_ndcg20, label='LightGCN++ NDCG@20')
plt.xlabel('Epochs')
plt.ylabel('NDCG@20')
plt.title('amazon book')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()