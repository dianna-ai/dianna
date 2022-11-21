#!/usr/bin/env python3

import torch
from models import CnnClassificationBaseline
from deeprank.learn import DataSet
import torch.utils.data as data_utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from time import perf_counter

from dianna.methods.rise import RISEImage
np.random.seed(42)

device_name = 'cuda'

DATA_PATH = './'
hdf5_path = DATA_PATH + '000_hla_drb1_0101_15mers.hdf5'
sample_path = DATA_PATH + 'one_sample.hdf5'
pretrained_model = 'best_valid_model.pth.tar'

data_set = DataSet(
    sample_path,
    chain1="M",
    chain2="P",
    process=False)

state = torch.load(pretrained_model,  map_location=device_name)

for key in ['select_feature', 'select_target', 'pair_chain_feature', 'dict_filter',
            'normalize_targets', 'normalize_features', 'transform', 'proj2D',
            'target_ordering', 'clip_features', 'clip_factor', 'mapfly', 'grid_info']:
    setattr(data_set, key, state[key])

if data_set.normalize_targets:
    data_set.target_min = state['target_min']
    data_set.target_max = state['target_max']

if data_set.normalize_features:
    data_set.feature_mean = state['feature_mean']
    data_set.feature_std = state['feature_std']

data_set.process_dataset()

net = CnnClassificationBaseline(data_set.input_shape)
device = torch.device(device_name)
net.to(device)
if state['cuda']:
    for paramname in list(state['state_dict'].keys()):
        paramname_new = paramname.lstrip('module.')
        if paramname != paramname_new:
            state['state_dict'][paramname_new] = state['state_dict'][paramname]
            del state['state_dict'][paramname]

net.load_state_dict(state['state_dict'])

index = list(range(len(data_set)))
sampler = data_utils.sampler.SubsetRandomSampler(index)
loader = data_utils.DataLoader(data_set, sampler=sampler)

net.train(mode=False)
torch.set_grad_enabled(False)

sample = next(iter(loader))
feature = sample['feature']
target = sample['target']

feature_dianna = feature.data.numpy()[0]
axis_labels = ['channels', 'x', 'y', 'z']


def run_model(data_item):
    outputs = net(data_item)
    tmp = outputs.to(torch.float32)
    return F.softmax(tmp, dim=1).data.cpu().numpy()


def prepare_input_data(data):
    return Variable(torch.tensor(data).to(device)).float()


# heatmaps = dianna.explain_image(run_model, feature_dianna, "RISE", axis_labels=axis_labels, preprocess_function=prepare_input_data, p_keep=.4, labels=(0,1))
n_masks = 1024
#n_masks = 32
#feature_res=8
feature_res=1
rise = RISEImage(n_masks=n_masks, feature_res=feature_res, p_keep=.4,
                 axis_labels=axis_labels, preprocess_function=prepare_input_data)

t1 = perf_counter()
heatmaps = rise.explain(run_model, feature_dianna, labels=(0, 1))
t2 = perf_counter()
print(f"Explaining took {t2-t1:.2f} seconds")

grid = np.indices(feature_dianna.shape[1:])


for idx, heatmap in enumerate(heatmaps):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(*grid, c=heatmap, alpha=.1)
    plt.savefig(f'heatmap_{idx}.png')
    plt.close(fig)

np.save('heatmaps', heatmaps)
