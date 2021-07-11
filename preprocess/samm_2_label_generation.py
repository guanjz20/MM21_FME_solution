'''
generate the emotion intensity of each frame
'''

# %%
import os
import pdb
import os.path as osp
from numpy.core.numeric import ones
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import params

# %% main
anno_dict = {}  # intensity
label_dict = {}  # 0: none, 1: macro, 2: micro
pred_gt = {}  # [[onset, offset, label],...]
bi_label_dict = {}  # store all the img_ps fall into the spotting interval
df = pd.read_csv(osp.join(params.SAMM_ROOT, 'SAMM_labels.csv'))
data = df.values
for row in data:
    # construct imgs dir for current row data
    file_name = row[1][:5]
    imgs_dir = osp.join(params.SAMM_VIDEO_DIR, file_name)
    assert osp.exists(imgs_dir)

    # update emotion intensity and label
    imgs_name = [
        name
        for name in sorted(os.listdir(imgs_dir),
                           key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if '.jpg' in name
    ]  # first img name: xxx_x_0001.jpg

    onset, apex, offset = row[3:3 + 3]
    onset, apex, offset = int(onset), int(apex), int(offset)
    if onset > 0 and apex > 0 and offset > 0:
        pass
    elif onset > 0 and apex > 0 and offset == -1:
        offset = min(len(imgs_name), apex + (apex - onset))
    elif onset > 0 and apex == -1 and offset > 0:
        apex = (onset + offset) // 2
    else:
        raise Exception

    try:
        assert onset < apex and apex < offset
    except:
        print('[Error][{}] onset: {}, apex: {}, offset: {}, '.format(
            imgs_dir, onset, apex, offset))
        continue  # skip this row

    if not imgs_dir in anno_dict:
        anno_dict[imgs_dir] = np.zeros(len(imgs_name))
        label_dict[imgs_dir] = np.zeros(len(imgs_name))
        pred_gt[imgs_dir] = []
        bi_label_dict[imgs_dir] = []

    # convert start index from 1 to 0
    onset -= 1
    apex -= 1
    offset -= 1

    # intensity
    sigma = min(offset - apex, apex - onset) // 2 + 1e-7
    if sigma <= 0:
        pdb.set_trace()

    mu = apex
    func = lambda x: np.exp(-(x - mu)**2 / 2 / sigma / sigma
                            ) / sigma / np.sqrt(2 * np.pi)
    cumsum = 0
    for i in range(onset, offset + 1):
        anno_dict[imgs_dir][i] += func(i)
        cumsum += anno_dict[imgs_dir][i]
    # print('onset2offset cumsum: {:.2f}'.format(cumsum))
    # label
    label_dict[imgs_dir][onset:offset +
                         1] = 1 if 'macro' in str(row[-2]).lower() else 2
    # pred_gt
    pred_gt[imgs_dir].append(
        [onset, offset + 1, 1 if 'macro' in str(row[-2]).lower() else 2])
    # bi_label
    bi_label_dict[imgs_dir].extend(
        [osp.join(imgs_dir, name) for name in imgs_name[onset:offset + 1]])

np.save(osp.join(params.SAMM_ROOT, 'anno_dict.npy'), anno_dict)
np.save(osp.join(params.SAMM_ROOT, 'label_dict.npy'), label_dict)
np.save(osp.join(params.SAMM_ROOT, 'pred_gt.npy'), pred_gt)
np.save(osp.join(params.SAMM_ROOT, 'bi_label.npy'), bi_label_dict)

# %% visulization
# fig = plt.figure(figsize=(30, 50))
# for i, (k, v) in enumerate(anno_dict.items()):
#     fig.add_subplot((len(anno_dict) - 1) // 5 + 1, 5, i + 1)
#     plt.plot(v)
# fig.tight_layout()
# plt.savefig('./SAMM_annos.pdf')
# plt.show()

column = 5
fig = plt.figure(figsize=(30, ((len(label_dict) - 1) // column + 1) * 2))
for i, (k, v) in enumerate(label_dict.items()):
    v[v > 0] = 1  # 1,2 -> 1
    fig.add_subplot((len(label_dict) - 1) // column + 1, column, i + 1)
    plt.plot(v, 'r-')
    plt.title(osp.basename(k))
fig.tight_layout()
out_dir = './preprocess'
plt.savefig(osp.join(out_dir, 'sa_bi_label.pdf'))
plt.close('all')

# %%
