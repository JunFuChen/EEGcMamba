# The following code is based on or inspired by the work of Xiang Zhang from the project https://github.com/mims-harvard/TFC-pretraining/tree/main

import numpy as np
import torch

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform_TD_bank(sample, augmentation):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = jitter(sample,  augmentation.jitter_ratio) 
    aug_2 = scaling(sample,  augmentation.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments= augmentation.max_seg) 
    aug_4 = masking(sample, keepratio=0.9) 
    aug_1 = jitter(sample, config.augmentation.jitter_ratio)
    aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0][:, None, None]
    aug_2 = aug_2 * li_onehot[:, 0][:, None, None]
    aug_3 = aug_3 * li_onehot[:, 0][:, None, None]
    aug_4 = aug_4 * li_onehot[:, 0][:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T



def DataTransform_TD(sample, config):
    """Simplely use the jittering augmentation. We noticed that in EEGcMamba, the augmentation has litter impact on the clustering performance."""
    aug = jitter(sample, config.augmentation.jitter_ratio)
    return aug


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return np.random.binomial(1, p, size=(B, T, D))

def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.shape[0], x.shape[1], x.shape[2], p=keepratio)
    x[~mask_id] = 0
    return x

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.5):
    # https://arxiv.org/pdf/1706.00527.pdf  # sample dimension Timestep
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :]) 
    return np.concatenate((ai), axis=1)



def permutation(x, max_segments=5, seg_mode="random"): 
    orig_steps = np.arange(x.shape[2]) 


    ret = np.zeros_like(x)
    for i, pat in enumerate(x): 
        if num_segs[i] > 1: 
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False) 
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            splits = np2list(splits)
            splits = np.asarray(splits, dtype=object)
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            warp = warp.astype(int)
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat 
    return ret

def np2list(nparraylist):
    out2list = []
    for j in range(len(nparraylist)):
        # print(nparraylist[j].shape)
        if nparraylist[j].shape[0] != 0:
            nparraylist[j] = nparraylist[j].astype(int)
            list_j = nparraylist[j].tolist()
            out2list.append(list_j)
    return out2list

