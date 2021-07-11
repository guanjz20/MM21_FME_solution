import pandas as pd
import numpy as np
import os.path as osp

dataset = 'CASME_2'
# dataset = 'SAMM'
submit_name = 'submit_{}.csv'.format(dataset)
result_dir_name = 'results'
submit_npy_name = 'match_regions_record_all.npy'
submit_id = 'done_exp_cls_ca_20210708-215035'


def convert_key(k, dataset):
    if dataset == 'CASME_2':
        k = osp.basename(k)[:7]
    elif dataset == 'SAMM':
        k = osp.basename(k)
    else:
        raise NotImplementedError
    return k


data = np.load(osp.join('.', result_dir_name, submit_id, 'output',
                        submit_npy_name),
               allow_pickle=True).item()

metric = {'TP': 0, 'FN': 0, 'FP': 0}
with open(submit_name, 'w') as f:
    if dataset == 'CASME_2':
        f.write('2\r\n')
    elif dataset == 'SAMM':
        f.write('1\r\n')
    else:
        raise NotImplementedError
    for k, v in data.items():
        k = convert_key(k, dataset)
        assert isinstance(v[0], list)
        for line in v:
            f.write(','.join([k, *[str(x) for x in line]]) + '\r\n')
            metric[line[-1]] += 1

precision = metric['TP'] / (metric['TP'] + metric['FP'])
recall = metric['TP'] / (metric['TP'] + metric['FN'])
f_score = 2 * precision * recall / (precision + recall)
print('TP: {}, FP: {}, FN: {}'.format(metric['TP'], metric['FP'],
                                      metric['FN']))
print('P: {:.4f}, R: {:.4f}, F: {:.4f}'.format(precision, recall, f_score))
