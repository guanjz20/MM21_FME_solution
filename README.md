# Solution of ACM MM21 FME challenge - spotting track
## Requirements
- python 3.9.5
- torch 1.8.0
- timm 0.4.12
- pytorch-benchmarks (clone from [github](https://github.com/albanie/pytorch-benchmarks))

## Run - CAS(ME)$^2$ as example

### Pretrianed Weights
1. Download the weights of resnet50 pretrained on VGGFACE2 and FER$+$ from [here](https://www.robots.ox.ac.uk/~albanie/pytorch-models.html).
2. Download the weights of our model pretrained on Aff-wild from [here](https://cloud.tsinghua.edu.cn/d/58af3b49570741ab82f3/). Alternatively, you can download the Aff-wild dataset and train the model by yourself.
3. Change the paths of pretrained model properly.

### Dataset and Pre-processing
1. Download and extract the dataset.
2. Download the csv files reorganized by us from [here](https://cloud.tsinghua.edu.cn/d/58af3b49570741ab82f3/).
3. Change the data paths properly.
4. Run `python preprocess/casme_2_label_generation.py` for label generation.
5. Run `python CNN_feature_extraction.py` for spatial feature extraction.

### Train with leave-one-subject-out cross-validaiton
```
nohup python -m torch.distributed.launch --nproc_per_node 2 main_cls.py --distributed --amp \
--dataset CASME_2 --snap exp_cls_ca --print-freq 50 --gpus 0,1 \
--data_option wt_diff --workers 8 --batch_size 8 --input_size 128 --length 64 --step 64 -L 12 \
--optim SGD --lr 1e-3 --lr_steps 3 5 --lr_decay_factor 0.3 \
--patience 10 --focal_alpha 1 14 --early_stop 3 --epochs 8 \
--load_pretrained /home/gjz/fmr_backbone/pretrained_models/wtdf1_wt_size:112_length:64_L:12/model/fold_0_best_loss.pth.tar \
> running_cls_ca.log 2>&1 &
```
