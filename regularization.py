from dataloader.hands17_loader import Hands17
from torch.utils.data import DataLoader
import numpy as np
import torch

pairs = [(1, 6),
       (6, 7),
       (7, 8),
       (2, 9),
       (9, 10),
       (10, 11),
       (3,  12),
       (12, 13),
       (13, 14),
       (4, 15),
       (15, 16),
       (16, 17),
       (5, 18),
       (18, 19),
       (19, 20)]
    
dataset = Hands17(
    './data/hands17',
    "train",
    img_size=128,
    aug_para=[10, 0.1, 120],
    cube=[300, 300, 300],
)

trainLoader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
        )

bone_lengths = [[ ] for _ in range(15)]

def calc_eu_dist(points, pair):
    A = points[:, pair[0], :]
    B = points[:, pair[1], :]
    squared_diff = (A - B) ** 2

    # Sum the squared differences along the coordinate axis (axis=1)
    sum_squared_diff = torch.sum(squared_diff, dim=1)

    # Take the square root to get the Euclidean distance
    distance = torch.sqrt(sum_squared_diff)

    return distance
    

for img, jt_xyz_gt, jt_uvd_gt, center_xyz, M, cube in trainLoader:
    for idx, pair in enumerate(pairs):
        distance = calc_eu_dist(jt_uvd_gt, pair)
        bone_lengths[idx].extend(distance.tolist())

bone_lengths = np.array(bone_lengths)
np.save('bone_len.npy', bone_lengths)


# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the data from the npy file
data = np.load('bone_len.npy')

# # Now 'data' contains the contents of the npy file
# print(data.shape)

# # Create a density plot
# sns.kdeplot(data[0]/data[1], fill=True, color='skyblue')
# plt.title('Density Plot of One-Dimensional Data')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.show()

regu_ratio = [[ ] for _ in range(10)]

pairs = [(0,1),
         (1,2),
         (3,4),
         (4,5),
         (6,7),
         (7,8),
         (9,10),
         (10,11),
         (12,13),
         (13,14)]

def find_2_lim(data_points):
    sorted_data = np.sort(data_points)
    return np.percentile(sorted_data, 0.5), np.percentile(sorted_data, 99.5)

# to find "normal" range of bone ratios!
for pair in pairs:
    ratio_data = data[pair[0]]/data[pair[1]]
    low, high = find_2_lim(ratio_data)
    print(f'{low}:{high}')    