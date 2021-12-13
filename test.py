import torch

val=[[0.5, 0.5, 0.5]]
n_boxes = len(val)
scale_range_min=[0.5, 0.5, 0.5]
scale_range_max=[0.6, 0.6, 0.6]
scale_range = (torch.tensor(scale_range_max) -
                            torch.tensor(scale_range_min)).reshape(1, 1, 3)
scale_min = torch.tensor(scale_range_min).reshape(1, 1, 3)
# print(scale_min)
s_rand = torch.rand(4, n_boxes, 1)
print(s_rand)
print(scale_range)
s = s_rand * scale_range
print(s)
