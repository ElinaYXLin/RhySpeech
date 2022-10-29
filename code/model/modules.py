import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad, to_device


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.pause_predictor = PausePredictor(model_config)
    #    self.pause_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        pause_bins = model_config["pause_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]
            pause_min, pause_max = stats["pause"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        self.pause_bins = nn.Parameter(
                torch.linspace(pause_min, pause_max, pause_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.pause_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def get_pause_embedding(self, x, corrText, target, mask, control):
        prediction = self.pause_predictor(x, mask, corrText)
        
        if target is not None:
            embedding = self.pause_embedding(torch.bucketize(target, self.pause_bins))
        else:
            prediction = prediction * control
            embedding = self.pause_embedding(
                torch.bucketize(prediction, self.pause_bins)
            )
        return prediction, embedding

    # def get_pause_embedding(self, x, target, mask, control):
    # # for using standard variance predictor
    #     prediction = self.pause_predictor(x, mask)
    #   #  
    #     if target is not None:
    #         embedding = self.pause_embedding(torch.bucketize(target, self.pause_bins))
    #     else:
    #         prediction = prediction * control
    #         embedding = self.pause_embedding(
    #             torch.bucketize(prediction, self.pause_bins)
    #         )
    #     return prediction, embedding

    def forward(
        self,
        x,
        corrText,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        pause_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        r_control=1.0,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        pause_prediction, pause_embedding = self.get_pause_embedding(
                x, corrText, pause_target, src_mask, r_control
        )
        # pause_prediction, pause_embedding = self.get_pause_embedding(
        #     x, pause_target, src_mask, r_control
        # )
     #   x = x + pause_embedding # coMMent out this line if doing "Network Structure"
    

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            pause_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class FlexSigmoid(torch.nn.Module):
    def __init__(self):
        super(FlexSigmoid, self).__init__()  # changed
        self.paramA = torch.nn.Parameter(torch.Tensor(1))
        self.paramA.data.uniform_(2.2,2.5)
        self.paramB = torch.nn.Parameter(torch.Tensor(1))
        self.paramB.data.uniform_(1.2, 1.5)
        self.paramC = torch.nn.Parameter(torch.Tensor(1))
        self.paramC.data.uniform_(2.2, 2.5)
        self.paramD = torch.nn.Parameter(torch.Tensor(1))
        self.paramD.data.uniform_(0, 0.1)

    def forward(self, x):
        res = torch.Tensor(1)
        if (self.paramC != 0):
            res = (self.paramA* self.paramB* x)/pow((1+pow(self.paramB*abs(x), self.paramC)), 1/self.paramC) + self.paramD*x
        else:
            res = self.paramD*x
        return res

class PausePredictor(nn.Module):
    """Pause Predictor"""

    def __init__(self, model_config):
        super(PausePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size+16,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", FlexSigmoid()),
                #    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                   ("relu_2", FlexSigmoid()),
                #    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask, textEmb):
       # full_textEmb = textEmb.unsqueeze(2).expand(-1, -1, encoder_output.size()[2])
        full_textEmb = torch.empty((encoder_output.size()[0], encoder_output.size()[1], 16)) # 16 elements before
        for i in range(encoder_output.size()[0]):
            for j in range(encoder_output.size()[1]):
                for k in range (16):
                    if (j-k >= 0):
                        full_textEmb[i][j][k] = textEmb[i][j-k]
                    else:
                        full_textEmb[i][j][k] = -1
        full_textEmb = full_textEmb.to(device)
        out = torch.cat((encoder_output.to(device), full_textEmb), 2)
        out = self.conv_layer(out)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask.to("cuda"), 0.0)

        return out


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", FlexSigmoid()),
                #    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", FlexSigmoid()),
                #    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
    #    VariancePredictor.requires_grad=False # only for fine-tuning
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask.to("cuda"), 0.0)

        return out



class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x.to("cuda"))
        x = x.contiguous().transpose(1, 2)

        return x


# class PauseAdaptor(nn.Module):
#     """Pause Adaptor"""

#     def __init__(self, preprocess_config, model_config):
#         super(PauseAdaptor, self).__init__()
#         self.pause_predictor = PausePredictor(model_config)

#         n_bins = model_config["pause_embedding"]["n_bins"]

#         self.pause_embedding = nn.Embedding(
#             n_bins, model_config["transformer"]["encoder_hidden"]
#         )

#         with open(
#             os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
#         ) as f:
#             stats = json.load(f)
#             pause_min, pause_max = stats["pause"][:2]

#         self.pause_bins = nn.Parameter(
#                 torch.linspace(pause_min, pause_max, n_bins - 1),
#                 requires_grad=False,
#             )

#     def get_pause_embedding(self, x, textEmb, target, mask, control):
#         prediction = self.pause_predictor(x, mask, textEmb)
#         if target is not None:
#             embedding = self.pause_embedding(torch.bucketize(target, self.pause_bins))
#         else:
#             prediction = prediction * control
#       #      prediction = torch.max(prediction, 0)
#             embedding = self.pause_embedding(
#                 torch.bucketize(prediction, self.pause_bins)
#             )
#         return prediction, embedding

 

#     def forward(
#         self,
#         x,
#         corrText,
#         src_mask,
#         pause_target=None,
#         r_control=1.0,
#     ):
#         # rhythm/pause
#         pause_prediction, pause_embedding = self.get_pause_embedding(
#                 x, corrText, pause_target, src_mask, r_control
#         )
    
#         x = x + pause_embedding

#         return (
#             x,
#             pause_prediction,
#         )
