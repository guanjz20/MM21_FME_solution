'''
generate the emotion intensity of each frame
'''

# %%
import pdb
import os
import os.path as osp
from numpy.core.numeric import ones, ones_like
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import params

# %% ID2NAME and NAME2ID

# CASME_2_PID2NAME/NAME2PID
df = pd.read_csv(osp.join(params.CASME_2_LABEL_DIR, 'naming_rule1.csv'))
data = df.values
CASME_2_PID2NAME = {str(line[-1]): str(line[1]) for line in data}
CASME_2_NAME2PID = {str(line[1]): str(line[-1]) for line in data}
del df
del data

# CASME_2_VID2NAME
df = pd.read_csv(osp.join(params.CASME_2_LABEL_DIR, 'naming_rule2.csv'))
data = df.values
CASME_2_VID2NAME = {'{:04d}'.format(line[0]): str(line[1]) for line in data}
CASME_2_NAME2VID = {str(line[1]): '{:04d}'.format(line[0]) for line in data}
del df
del data

save_dict_dir = osp.join(params.CASME_2_ROOT, 'ID2NAME2ID')
os.makedirs(save_dict_dir, exist_ok=True)
for p, d in zip(
    ['pid2name', 'name2pid', 'vid2name', 'name2vid'],
    [CASME_2_PID2NAME, CASME_2_NAME2PID, CASME_2_VID2NAME, CASME_2_NAME2VID]):
    np.save(osp.join(save_dict_dir, p + '.npy'), d)

# %% main
anno_dict = {}
label_dict = {}  # 0: none, 1: macro, 2: micro
pred_gt = {}  # [[onset, offset, label],...]
bi_label_dict = {}  # store all the img_ps fall into the spotting interval
df = pd.read_csv(osp.join(params.CASME_2_LABEL_DIR, 'CASFEcode_final.csv'))
data = df.values
for row in data:
    # construct imgs dir for current row data
    pid = str(row[0])
    vname = row[1].split('_')[0]
    pname = CASME_2_PID2NAME[pid]
    vid = CASME_2_NAME2VID[vname]
    name_code = pname[1:]
    imgs_file_head = name_code + '_' + vid
    for file_name in os.listdir(osp.join(params.CASME_2_VIDEO_DIR, pname)):
        if file_name.startswith(imgs_file_head):
            imgs_dir = osp.join(params.CASME_2_VIDEO_DIR, pname, file_name)
            break

    # update emotion intensity and label
    imgs_name = [
        name
        for name in sorted(os.listdir(imgs_dir),
                           key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if '.jpg' in name
    ]  # first img name: img_1.jpg

    onset, apex, offset = row[2:2 + 3]
    onset, apex, offset = int(onset), int(apex), int(offset)
    if onset > 0 and apex > 0 and offset > 0:
        pass
    elif onset > 0 and apex > 0 and offset == 0:
        offset = min(len(imgs_name), apex + (apex - onset))
    elif onset > 0 and apex == 0 and offset > 0:
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
    sigma = min(offset - apex, apex - onset) // 2
    mu = apex
    func = lambda x: np.exp(-(x - mu)**2 / 2 / sigma / sigma
                            ) / sigma / np.sqrt(2 * np.pi)
    # func = lambda x: (x - onset) / (apex - onset) if x >= apex else (
    #     offset - x) / (offset - apex)

    cumsum = 0
    for i in range(onset, offset + 1):
        anno_dict[imgs_dir][i] += func(i)
        cumsum += anno_dict[imgs_dir][i]
    if cumsum < 0:
        pdb.set_trace()
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

np.save(osp.join(params.CASME_2_LABEL_DIR, 'anno_dict.npy'), anno_dict)
np.save(osp.join(params.CASME_2_LABEL_DIR, 'label_dict.npy'), label_dict)
np.save(osp.join(params.CASME_2_LABEL_DIR, 'pred_gt.npy'), pred_gt)
np.save(osp.join(params.CASME_2_LABEL_DIR, 'bi_label.npy'), bi_label_dict)

# %% visulization
# fig = plt.figure(figsize=(30, 50))
# for i, (k, v) in enumerate(anno_dict.items()):
#     fig.add_subplot((len(anno_dict) - 1) // 5 + 1, 5, i + 1)
#     plt.plot(v)
# fig.tight_layout()
# plt.savefig('./CASME_2_annos.pdf')
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
plt.savefig(osp.join(out_dir, 'ca_bi_label.pdf'))
plt.close('all')

# %%
