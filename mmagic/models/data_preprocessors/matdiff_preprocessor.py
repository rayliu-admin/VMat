# Copyright (c) OpenMMLab. All rights reserved.

from logging import WARNING
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine import print_log

import numpy as np
from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from .data_preprocessor import DataPreprocessor

DataSamples = Optional[Union[list, torch.Tensor]]
ForwardResults = Union[Dict[str, torch.Tensor], List[DataSample],
                       Tuple[torch.Tensor], torch.Tensor]
MEAN_STD_TYPE = Union[Sequence[Union[float, int]], float, int]


@MODELS.register_module()
class MatdiffPreprocessor(DataPreprocessor):
    """DataPreprocessor for matting models.

    See base class ``DataPreprocessor`` for detailed information.

    Workflow as follow :

    - Collate and move data to the target device.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Stack inputs to batch_inputs.

    Args:
        mean (Sequence[float or int], float or int, optional): The pixel mean
            of image channels. Noted that normalization operation is performed
            *after channel order conversion*. If it is not specified, images
            will not be normalized. Defaults None.
        std (Sequence[float or int], float or int, optional): The pixel
            standard deviation of image channels. Noted that normalization
            operation is performed *after channel order conversion*. If it is
            not specified, images will not be normalized. Defaults None.
        proc_trimap (str): Methods to process gt tensors.
            Default: 'rescale_to_zero_one'.
            Available options are ``rescale_to_zero_one`` and ``as-is``.
        stack_data_sample (bool): Whether stack a list of data samples to one
            data sample. Only support with input data samples are
            `DataSamples`. Defaults to True.
    """

    def __init__(self,
                 mean: MEAN_STD_TYPE = [123.675, 116.28, 103.53],
                 std: MEAN_STD_TYPE = [58.395, 57.12, 57.375],
                 output_channel_order: str = 'RGB',
                 proc_trimap: str = 'rescale_to_zero_one',
                 num_timesteps = 6,
                 stack_data_sample=True,
                 diffusion_cfg = None):
        # specific data_keys for matting task
        data_keys = ['gt_fg', 'gt_bg', 'gt_merged', 'gt_alpha']
        super().__init__(
            mean,
            std,
            output_channel_order=output_channel_order,
            data_keys=data_keys,
            stack_data_sample=stack_data_sample)

        self.proc_trimap = proc_trimap
        self.num_timesteps = num_timesteps
        self._diffusion_init(diffusion_cfg)
        # self.proc_gt = proc_gt

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


    def _proc_batch_trimap(self, batch_trimaps: torch.Tensor):

        if self.proc_trimap == 'rescale_to_zero_one':
            batch_trimaps = batch_trimaps / 255.0  # uint8->float32
        elif self.proc_trimap == 'as_is':
            batch_trimaps = batch_trimaps.to(torch.float32)
        else:
            raise ValueError(
                f'proc_trimap = {self.proc_trimap} is not supported.')

        return batch_trimaps

    def _preprocess_data_sample(self, data_samples: SampleList,
                                training: bool) -> list:
        """Preprocess data samples. When `training` is True, fields belong to
        :attr:`self.data_keys` will be converted to
        :attr:`self.output_channel_order` and *divided by 255*. When `training`
        is False, fields belongs to :attr:`self.data_keys` will be attempted
        to convert to 'BGR' without normalization. The corresponding metainfo
        related to normalization, channel order conversion will be updated to
        data sample as well.

        Args:
            data_samples (List[DataSample]): A list of data samples to
                preprocess.
            training (bool): Whether in training mode.

        Returns:
            list: The list of processed data samples.
        """
        if not training:
            # set default order to BGR in test stage
            target_order = 'BGR'
        else:
            # conversion as default (None)
            target_order = self.output_channel_order

        for data_sample in data_samples:
            for key in self.data_keys:
                if not hasattr(data_sample, key):
                    # do not raise error here
                    if key != 'gt_fg' and not training:
                        # gt_fg is not required in test stage, therefore do
                        # not print log
                        print_log(f'Cannot find key \'{key}\' in data sample.',
                                  'current', WARNING)
                    break

                data = data_sample.get(key)
                data_channel_order = self._parse_channel_order(
                    key, data, data_sample)
                data, channel_order = self._do_conversion(
                    data, data_channel_order, target_order)
                if training:
                    data = data / 255.  # NOTE: divided by 255
                data_sample.set_data({f'{key}': data})
                data_process_meta = {
                    f'{key}_enable_norm': self._enable_normalize,
                    f'{key}_output_channel_order': channel_order,
                    f'{key}_mean': self.mean,
                    f'{key}_std': self.std
                }
                data_sample.set_metainfo(data_process_meta)

        if self.stack_data_sample:
            return DataSample.stack(data_samples)

        return data_samples
    
    def uniform_sampler(self,num_steps, batch_size, device):
        all_indices = np.arange(num_steps)
        indices_np = np.random.choice(all_indices, size=(batch_size,))
        indices = torch.from_numpy(indices_np).long().to(device)
        return indices  
    
    def q_sample(self, x_start, x_last, t, current_device):
        q_ori_probs = torch.tensor(self.betas_cumprod, device=current_device)
        q_ori_probs = q_ori_probs[t]
        q_ori_probs = q_ori_probs.reshape(-1, 1, 1, 1)
        sample_noise = torch.rand(size=x_start.shape, device=current_device)
        transition_map = (sample_noise < q_ori_probs).float()
        sample = transition_map * x_start + (1 - transition_map) * x_last
        return sample

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, list]:
        """Pre-process input images, trimaps, ground-truth as configured.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Tuple[torch.Tensor, list]:
                Batched inputs and list of data samples.
        """
        if not training:
            # Image may of different size when testing
            assert len(data['data_samples']) == 1, (
                'only batch_size=1 is supported for testing.')
        data = super().forward(data, training=training)

        batch_images = data['inputs']
        # batch_trimaps = data['data_samples'].trimap
        # batch_trimaps = self._proc_batch_trimap(batch_trimaps)

        # Stack image and trimap along channel dimension
        # All existing models do concat at the start of forwarding
        # and data_sample is a very complex data structure
        # so this is a simple work-around to make codes simpler
        # print(f"batch_trimap.dtype = {batch_trimap.dtype}")

        # assert batch_images.ndim == batch_trimaps.ndim == 4
        # assert batch_images.shape[-2:] == batch_trimaps.shape[-2:], (
        #     'Expect merged.shape[-2:] == trimap.shape[-2:], '
        #     f'but got {batch_images.shape[-2:]} vs {batch_trimaps.shape[-2:]}')


        # diffusion preprocessor
        current_device = batch_images.device
        coarse_mask = data['data_samples'].coarse_masks
        gt_alpha = data['data_samples'].gt_alpha
        t = self.uniform_sampler(self.num_timesteps, batch_images.shape[0], current_device)
        x_t = self.q_sample(gt_alpha, coarse_mask, t, current_device)

        # N, (4/6), H, W
        batch_inputs = torch.cat((batch_images, x_t), dim=1)
        
        # TODO: need pass t to model
        data['data_samples'].t = t
        data['inputs'] = batch_inputs.float()
        return data
