import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
import os.path as osp

from torch.serialization import load


class MLP(nn.Module):
    def __init__(self, hidden_units, dropout=0.3):
        super(MLP, self).__init__()
        input_feature_dim = hidden_units[0]
        num_layers = len(hidden_units) - 1
        assert num_layers > 0
        assert hidden_units[-1] == 256
        fc_list = []
        for hidden_dim in hidden_units[1:]:
            fc_list += [
                nn.Dropout(dropout),
                nn.Linear(input_feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ]
            input_feature_dim = hidden_dim
        self.mlp = nn.Sequential(*fc_list)

    def forward(self, input_tensor):
        bs, num_frames, feature_dim = input_tensor.size()
        input_tensor = input_tensor.reshape(bs * num_frames, feature_dim)
        out = self.mlp(input_tensor)
        return out.reshape(bs, num_frames, -1)


class Temporal_Net(nn.Module):
    def __init__(self, input_size, num_channels, hidden_units, dropout,
                 feature):
        super().__init__()
        assert input_size in [112, 128, 224, 256]
        self.feature = feature  # return feature before classification

        # 4 layers conv net
        self.conv_net = []
        self.conv_net.append(
            self._make_conv_layer(num_channels, 2**6, stride=2))
        for i in range(7, 10):
            self.conv_net.append(
                self._make_conv_layer(2**(i - 1), 2**i, stride=2))
        self.conv_net = nn.Sequential(*self.conv_net)

        last_conv_width = input_size // (2**4)
        last_conv_dim = 2**9
        self.dropout = nn.Dropout2d(p=0.2)
        # self.avgpool = nn.AvgPool2d(
        #     kernel_size=[last_conv_width, last_conv_width])
        fc_list = []
        fc_list += [
            nn.Linear(last_conv_dim, hidden_units[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_units[0]),
            nn.Dropout(dropout)
        ]
        for i in range(0, len(hidden_units) - 2):
            fc_list += [
                nn.Linear(hidden_units[i], hidden_units[i + 1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_units[i + 1]),
                nn.Dropout(dropout)
            ]
        self.fc = nn.Sequential(*fc_list)

        # not used
        final_norm = nn.BatchNorm1d(1, eps=1e-6, momentum=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_units[-2], hidden_units[-1]), final_norm)

    def _make_conv_layer(self, in_c, out_c, kernel_size=3, stride=2):
        ks = kernel_size
        conv_layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(ks, ks), padding=ks // 2),
            nn.BatchNorm2d(out_c,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,
                      out_c,
                      kernel_size=(ks, ks),
                      padding=ks // 2,
                      stride=stride),
            nn.BatchNorm2d(out_c,
                           eps=1e-05,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        return conv_layer

    def forward(self, wt_data):
        bs, num_frames, num_channel, W0, H0 = wt_data.size()
        wt_data = wt_data.reshape(bs * num_frames, num_channel, W0, H0)
        conv_out = self.conv_net(wt_data)
        avgpool = F.adaptive_avg_pool2d(conv_out, (1, 1))
        # avgpool = self.avgpool(conv_out)
        avgpool = avgpool.reshape(bs * num_frames, -1)
        out = self.fc(avgpool)
        if self.feature:
            return out
        else:
            out = self.classifier(out)
            return out


class Two_Stream_RNN(nn.Module):
    def __init__(self,
                 mlp_hidden_units=[2048, 256, 256],
                 dropout=0.3,
                 inchannel=12,
                 size=256,
                 outchannel=4):
        super().__init__()
        self.mlp = MLP(mlp_hidden_units)
        self.temporal_net = Temporal_Net(size,
                                         inchannel,
                                         hidden_units=[256, 256, 1],
                                         dropout=0.3,
                                         feature=True)

        self.transform = nn.Sequential(nn.Linear(512, 256),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm1d(256),
                                       nn.Dropout(dropout))
        self.rnns = nn.GRU(256,
                           128,
                           bidirectional=True,
                           num_layers=2,
                           dropout=0.3,
                           batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(256, outchannel),
                                        nn.BatchNorm1d(outchannel), nn.ReLU())
        _init_weights(self)

    def forward(self, temp_data, rgb_data, return_feature=False):
        bs, num_frames = rgb_data.size(0), rgb_data.size(1)

        # spatial features
        features_cnn = self.mlp(rgb_data)
        features_spatial = features_cnn.reshape(bs, num_frames, -1)

        # temporal features
        features_temporal = self.temporal_net(temp_data)
        features_temporal = features_temporal.reshape(bs, num_frames, -1)
        features = torch.cat([features_spatial, features_temporal], dim=-1)
        features = self.transform(features.reshape(bs * num_frames, -1))
        features = features.reshape(bs, num_frames, -1)

        # rnn combination
        outputs_rnns, _ = self.rnns(features)
        outputs_rnns = outputs_rnns.reshape(bs * num_frames, -1)
        out = self.classifier(outputs_rnns)
        out = out.reshape(bs, num_frames, -1)

        if return_feature:
            return out

        # anno transforms
        out[..., 0] = torch.log(out[..., 0] + 1)
        return out


class Two_Stream_RNN_Cls(Two_Stream_RNN):
    def __init__(self,
                 mlp_hidden_units=[2048, 256, 256],
                 dropout=0.3,
                 inchannel=12,
                 size=256,
                 outchannel=2):
        super().__init__(mlp_hidden_units=mlp_hidden_units,
                         dropout=dropout,
                         inchannel=inchannel,
                         size=size,
                         outchannel=outchannel)

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(256, outchannel))
        _init_weights(self)

    def forward(self, temp_data, rgb_data):
        out = super().forward(temp_data, rgb_data, return_feature=True)
        return out


class ResNet50_Cls(nn.Module):
    def __init__(self, num_class=2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.Dropout(0.5),
                                nn.Linear(512, num_class))

    def forward(self, x):
        assert x.shape[-1] == 2048
        x = self.fc(x)
        return x


def _init_weights(model):
    for k, m in model.named_modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def load_pretrained_model(model, path, load_bn):
    model_dict = model.state_dict()
    state_dict = torch.load(path, map_location='cpu')['state_dict']
    state_dict = {
        k.replace('wt_net', 'temporal_net', 1): v
        for k, v in state_dict.items()
    }

    # bn filter
    if not load_bn:
        bn_keys = []
        for k in state_dict.keys():
            if 'running_mean' in k:
                bn_name = '.'.join(k.split('.')[:-1])
                for name in [
                        'weight', 'bias', 'running_mean', 'running_var',
                        'num_batches_tracked'
                ]:
                    bn_keys.append(bn_name + '.' + name)
        state_dict = {k: v for k, v in state_dict.items() if k not in bn_keys}

        # # module name rank adjust
        # for k, v in state_dict.items():
        #     if 'mlp.mlp.5' in k:
        #         state_dict[k.replace('mlp.mlp.5', 'mlp.mlp.4')] = v
        #         del state_dict[k]
        #     if 'temporal_net.fc.4' in k:
        #         state_dict[k.replace('temporal_net.fc.4',
        #                              'temporal_net.fc.3')] = v
        #         del state_dict[k]

    # classifier filter
    state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
