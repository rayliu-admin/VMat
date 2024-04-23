# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmagic.registry import TRANSFORMS

import random
import math
import cv2

# def _poly2mask(self, mask_ann, img_h, img_w):
#         """Private function to convert masks represented with polygon to
#         bitmaps.

#         Args:
#             mask_ann (list | dict): Polygon mask annotation input.
#             img_h (int): The height of output mask.
#             img_w (int): The width of output mask.

#         Returns:
#             numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
#         """

#         if isinstance(mask_ann, list):
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#             rle = maskUtils.merge(rles)
#         elif isinstance(mask_ann['counts'], list):
#             # uncompressed RLE
#             rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
#         else:
#             # rle
#             rle = mask_ann
#         mask = maskUtils.decode(rle)
#         return mask


@TRANSFORMS.register_module()
class LoadCoarseMasks(BaseTransform):
    def __init__(self,
                 test_mode=False):
        super().__init__()
        self.test_mode = test_mode

    def transform(self, results):
        if not self.test_mode:
            alpha = results['alpha']
            gt_mask = np.zeros(alpha.shape, dtype=np.uint8)
            gt_mask[alpha >= 128] = 255
            gt_mask[alpha <= 128] = 0
            coarse_masks = modify_boundary(gt_mask.copy())
            results['coarse_masks'] = coarse_masks
            results['fine_masks'] = gt_mask
        else:
            print('todo: imread coarse mask')
            # coarse_masks = results['coarse_info']['masks']

            # new_coarse_masks = []
            # for mask in coarse_masks:
            #     new_coarse_masks.append(self._poly2mask(mask, h, w))
            # results['coarse_masks'] = BitmapMasks(new_coarse_masks, h, w)
        # results['mask_fields'].append('coarse_masks')

        return results
       
    def __repr__(self):
        return self.__class__.__name__ + (
            f'(img_key={repr(self.test_mode)}, ')


def get_random_structure(size):
    # The provided model is trained with 
    #   choice = np.random.randint(4)
    # instead, which is a bug that we fixed here
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)

    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):
    # modifies boundary of the given mask.
    # remove consecutive vertice of the boundary by regional sample rate
    # ->
    # remove any vertice by sample rate
    # ->
    # move vertice by distance between vertice and center of the mask by move rate. 
    # input: np array of size [H,W] image
    # output: same shape as input
    
    # get boundaries
    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #only modified contours is needed actually. 
    sampled_contours = []   
    modified_contours = [] 

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)

        #remove region of contour
        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)
        
        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])
            
        idx_dist = sorted(idx_dist, key=lambda x:x[1])
        
        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]
        
       #remove_start = random.randrange(0, number_of_vertices-number_of_removes, 1)
        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour
        

        #sample contours
        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])

            #modify contours
            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate) # 0.1 means change position of vertex to 10 percent farther from center
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)
        
    #draw boundary
    gt = np.copy(image)
    image = np.zeros((image.shape[0], image.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    if len(image.shape) == 3:
        image = image[:, :, 0]
    image = perturb_seg(image, iou_target)

    image = image / 255
    image = (image >= 0.5).astype(np.uint8)
    
    return image


@TRANSFORMS.register_module()
class GetMaskedImage(BaseTransform):
    """Get masked image.

    Args:
        img_key (str): Key for clean image. Default: 'gt'.
        mask_key (str): Key for mask image. The mask shape should be
            (h, w, 1) while '1' indicate holes and '0' indicate valid
            regions. Default: 'mask'.
        img_key (str): Key for output image. Default: 'img'.
        zero_value (float): Pixel value of masked area.
    """

    def __init__(self,
                 img_key='gt',
                 mask_key='mask',
                 out_key='img',
                 zero_value=127.5):
        self.img_key = img_key
        self.mask_key = mask_key
        self.out_key = out_key
        self.zero_value = zero_value

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clean_img = results[self.img_key]  # uint8
        mask = results[self.mask_key]  # uint8

        masked_img = clean_img * (1.0 - mask) + self.zero_value * mask
        masked_img = masked_img.astype(np.float32)
        results[self.out_key] = masked_img

        # copy metainfo
        if f'ori_{self.img_key}_shape' in results:
            results[f'ori_{self.out_key}_shape'] = deepcopy(
                results[f'ori_{self.img_key}_shape'])
        if f'{self.img_key}_channel_order' in results:
            results[f'{self.out_key}_channel_order'] = deepcopy(
                results[f'{self.img_key}_channel_order'])
        if f'{self.img_key}_color_type' in results:
            results[f'{self.out_key}_color_type'] = deepcopy(
                results[f'{self.img_key}_color_type'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(img_key={repr(self.img_key)}, '
            f'mask_key={repr(self.mask_key)}, '
            f'out_key={repr(self.out_key)}, '
            f'zero_value={repr(self.zero_value)})')
