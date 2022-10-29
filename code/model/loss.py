from cmath import nan
import torch
import torch.nn as nn

from utils.tools import to_device
device = torch.device("cuda")

class StepLoss(torch.nn.Module):
    def __init__(self):
        super(StepLoss, self).__init__()

    def forward(self, x, tar):
        res1 = abs(x-tar)
        x1 = x - tar
        x2 = abs(x1)

        res2 = res1 + 0.25*torch.max(x2, torch.full(x.size(), 0.3).cuda())
        x3 = x2 - torch.max(x2, torch.full(x.size(), 0.3).cuda())

        res3  = res2 + 2.125*torch.max(x3, torch.full(x.size(), 0.4).cuda())
        x4 = x3 - torch.max(x, torch.full(x.size(), 0.4).cuda())

        res4 = res3 + 0.25*torch.max(x4, torch.full(x.size(), 0.3).cuda())

        finalRes = sum(res4)
        finalRes /= res4.size()[0]

        return finalRes

    def backward(x, grad_output):
        res, = x.saved_tensors
        return grad_output*res

class StepLossNew(torch.nn.Module):
    def __init__(self):
        super(StepLossNew, self).__init__()

    def forward(self, x, tar):
        x = x.to(device)
        tar = tar.to(device)
        res1 = abs(x-tar)
        x1 = x - tar
        x2 = abs(x1)

        res2 = res1 + 0.25*torch.max(x2, torch.full(x.size(), 0.3).to(device))
        x3 = x2 - torch.max(x2, torch.full(x.size(), 0.3).to(device))

        res3  = res2 + 2.125*torch.max(x3, torch.full(x.size(), 0.4).to(device))
        x4 = x3 - torch.max(x, torch.full(x.size(), 0.4).to(device))

        res4 = res3 + 0.25*torch.max(x4, torch.full(x.size(), 0.3).to(device))

    #    res4 += torch.max(0*x, -x * 10)

        finalRes = sum(res4)
        finalRes /= res4.size()[0]

        return finalRes

    def backward(x, grad_output):
        res, = x.saved_tensors
        return grad_output*res



class SparseLoss(torch.nn.Module):
    def __init__(self, pWeight=1, pMargin=0.5):
        super(SparseLoss, self).__init__()
        self.pw = pWeight # penalty size
        self.pm = pMargin # penalizing margin, what would be rounded to zero would be treated as zero

    def forward(self, x, tar):
        res = 0.0
        res += self.pm * (self.pw*torch.logical_and(torch.gt(0*x, x-self.pm), torch.logical_not(torch.eq(tar, 0*tar))))

        finalRes = sum(res)
        finalRes /= res.size()[0]
        return finalRes




class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.step_loss = StepLossNew()
        self.sparse_loss = SparseLoss()
        self.cel_loss = nn.CrossEntropyLoss() 

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            pause_targets,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            pause_predictions,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        pause_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)
        log_duration_predictions = log_duration_predictions.to("cuda").masked_select(src_masks.to("cuda"))
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions.to(device), mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions.to(device), mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
    #    pause_loss_middle = self.step_loss(pause_predictions.to(device), pause_targets)
        pause_loss_middle = (pause_predictions - pause_targets)* (pause_predictions - pause_targets)
        pause_loss_penalty = self.sparse_loss(pause_predictions, pause_targets)
        pause_loss = (sum(sum(pause_loss_middle))/(pause_loss_middle.size()[0]*pause_loss_middle.size()[1]) + 100* sum(pause_loss_penalty))/pause_loss_middle.size()[1]
    #    pause_loss = pause_loss * 0.25 # approx. 0.5 out of 4
    #    pause_loss = self.cel_loss(pause_predictions.to(device), pause_targets)
    #    pause_loss = self.mse_loss(pause_predictions, pause_targets)
    #    pause_loss += sum(pause_loss_penalty/pause_loss_penalty.size()[0]) * 10 # just tentative (originally is 1)
        pause_weight = 0.7
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + pause_loss*pause_weight
        )


        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            pause_loss*pause_weight
        )
