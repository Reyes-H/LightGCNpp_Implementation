import os

input_path = "data/ml-1m/ratings.dat"
output_dir = "data/ml-1m/"
train_file = os.path.join(output_dir, "train.txt")
valid_file = os.path.join(output_dir, "valid.txt")
test_file = os.path.join(output_dir, "test.txt")

# 读取数据
interactions = []
with open(input_path, "r") as f:
    for line in f:
        user, item, rating, timestamp = line.strip().split("::")
        interactions.append((int(user), int(item), int(rating), int(timestamp)))

# 按用户分组，做简单的留一法划分
from collections import defaultdict
import random

user_hist = defaultdict(list)
for u, i, r, t in interactions:
    user_hist[u].append((t, i))

with open(train_file, "w") as f_train, open(valid_file, "w") as f_valid, open(test_file, "w") as f_test:
    for u in user_hist:
        items = sorted(user_hist[u], key=lambda x: x[0])  # 按时间排序
        if len(items) < 3:
            for _, i in items:
                f_train.write(f"{u} {i}\n")
        else:
            for _, i in items[:-2]:
                f_train.write(f"{u} {i}\n")
            f_valid.write(f"{u} {items[-2][1]}\n")
            f_test.write(f"{u} {items[-1][1]}\n")
print("Preprocessing done! train.txt, valid.txt, test.txt generated.")