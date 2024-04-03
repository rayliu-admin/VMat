from collections import OrderedDict
import torch
import torch.distributed as dist
import numpy as np
from abc import ABCMeta

from typing import Dict, List, Optional, Tuple, Union
from mmagic.models.base_models import BaseMattor
from mmagic.models.utils import get_unknown_tensor
from mmagic.registry import MODELS

from mmagic.structures import DataSample

DataSamples = Optional[Union[list, torch.Tensor]]

@MODELS.register_module()
class MATREFINER(BaseMattor):
    """Guided Contextual Attention image matting model.

    https://arxiv.org/abs/2001.04069

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        backbone (dict): Config of backbone.
        loss_alpha (dict): Config of the alpha prediction loss. Default: None.
        init_cfg (dict, optional): Initialization config dict. Default: None.
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``train_backbone`` should be specified. If the model has a refiner,
            ``train_refiner`` should be specified.
        test_cfg (dict): Config of testing. In ``test_cfg``, If the model has a
            refiner, ``train_refiner`` should be specified.
    """

    def __init__(self,
                 step,
                 diffusion_cfg,
                 data_preprocessor,
                 backbone,
                 loss_alpha=None,
                 init_cfg: Optional[dict] = None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(
            backbone=backbone,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

        self._diffusion_init(diffusion_cfg)
        self.step = step
        self.loss_alpha = MODELS.build(loss_alpha)

    def _diffusion_init(self, diffusion_cfg):
        self.diff_iter = diffusion_cfg['diff_iter']
        betas = diffusion_cfg['betas']
        self.eps = 1.e-6
        self.betas_cumprod = np.linspace(
            betas['start'], betas['stop'], 
            betas['num_timesteps'])
        betas_cumprod_prev = self.betas_cumprod[:-1]
        self.betas_cumprod_prev = np.insert(betas_cumprod_prev, 0, 1)
        self.betas = self.betas_cumprod / self.betas_cumprod_prev
        self.num_timesteps = self.betas_cumprod.shape[0]

    def forward(self,
                inputs: torch.Tensor,
                data_samples: DataSamples = None,
                mode: str = 'tensor') -> List[DataSample]:
        if mode == 'tensor':
            t = data_samples.t
            raw = self._forward(inputs,t)
            return raw
        elif mode == 'predict':
            # Pre-process runs in runner
            inputs = self.resize_inputs(inputs)
            batch_pred_alpha = self._forward_test(inputs)
            predictions = self.postprocess(batch_pred_alpha, data_samples)
            predictions = self.convert_to_datasample(predictions, data_samples)
            return predictions
        elif mode == 'loss':
            loss = self._forward_train(inputs, data_samples)
            return loss
        else:
            raise ValueError('Invalid forward mode.')
    
    def _forward(self, inputs, timesteps):
        """Forward function.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        raw_alpha = self.backbone(inputs, timesteps)
        return raw_alpha

    def p_sample(self, model_input, cur_fine_probs, t):
        pred_logits = self._forward(model_input, t)
        t = t[0].item()
        x_start_fine_probs = 2 * torch.abs(pred_logits.sigmoid() - 0.5)
        beta_cumprod = self.betas_cumprod[t]
        beta_cumprod_prev = self.betas_cumprod_prev[t]
        p_c_to_f = x_start_fine_probs * (beta_cumprod_prev - beta_cumprod) / (1 - x_start_fine_probs*beta_cumprod)
        cur_fine_probs = cur_fine_probs + (1 - cur_fine_probs) * p_c_to_f
        return pred_logits, cur_fine_probs

    def _forward_test(self, inputs, data_samples, use_last_step=True):
        """Forward function for testing GCA model.

        Args:
            inputs (torch.Tensor): batch input tensor.

        Returns:
            Tensor: Output tensor of model.
        """
        indices = list(range(self.num_timesteps))[::-1]
        indices = indices[:self.step]
        current_device = inputs.device
        x_last = data_samples.coarse_mask
        for i in indices:
            t = torch.tensor([i] * x.shape[0], device=current_device)
            last_step_flag = (use_last_step and i==indices[-1])
            model_input = torch.cat((inputs, x), dim=1)
            x, cur_fine_probs = self.p_sample(model_input, cur_fine_probs, t)

            if last_step_flag:
                x =  x.sigmoid()
            else:
                sample_noise = torch.rand(size=x.shape, device=x.device)
                fine_map = (sample_noise < cur_fine_probs).float()
                pred_x_start = (x >= 0).float()
                x = pred_x_start * fine_map + x_last * (1 - fine_map)

        return x

    def _forward_train(self, inputs, data_samples):
        """Forward function for training GCA model.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement]): data samples collated by
                :attr:`data_preprocessor`.

        Returns:
            dict: Contains the loss items and batch information.
        """
        gt_alpha = data_samples.gt_alpha
        t = data_samples.t
        pred_alpha = self._forward(inputs,t)

        # FormatTrimap(to_onehot=False) will change unknown_value to 1
        # FormatTrimap(to_onehot=True) will shift to 3 dim,
        # get_unknown_tensor can handle that directly without knowing
        # unknown_value.
        # weight = get_unknown_tensor(trimap, unknown_value=1)

        losses = {'loss': self.loss_alpha(pred_alpha, gt_alpha)}
        return losses


    

    