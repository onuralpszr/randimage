import random
from typing import Tuple

import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter


class BaseMask(object):
    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape = shape


class SaltPepperMask(BaseMask):
    def get_mask(self):
        return np.random.randint(0, 2, size=self.shape)


class NormalMask(BaseMask):
    def get_mask(self):
        mask = np.random.normal(0, 1, size=self.shape)
        return mask - np.min(mask)


class GaussianBlobMask(BaseMask):
    def get_mask(self, ncenters=None, sigma=None):
        mask = np.zeros(self.shape)
        if ncenters is None:
            ncenters = random.randint(1, int(0.5 * np.sqrt(mask.size)))
        if sigma is None:
            sigma = random.randint(1, int(0.2 * np.sqrt(mask.size)))
        for center in range(ncenters + 1):
            cx = random.randint(0, self.shape[0] - 1)
            cy = random.randint(0, self.shape[1] - 1)
            mask[cx, cy] = 1
        return gaussian_filter(mask, sigma, mode="nearest")

    def _get_gaussian_bell(self, center, sigma):
        return lambda point: multivariate_normal(center, sigma * np.eye(2)).pdf(point)

    # TODO Commented due to no use
    # def get_mask_slow(self, ncenters=None):
    #     if ncenters is None:
    #         ncenters = random.randint(1, int(0.5 * np.sqrt(self.mask.size)))
    #     self.mask = np.zeros(self.shape)
    #     gaussians = []
    #     for center in range(ncenters + 1):
    #         cx = random.randint(0, self.shape[0])
    #         cy = random.randint(0, self.shape[1])
    #         sigma = random.randint(1, int(0.2 * np.sqrt(self.mask.size)))
    #         gaussians.append(self._get_gaussian_bell((cx, cy), sigma))
    #     for idx, _ in np.ndenumerate(self.mask):
    #         self.mask[idx] = sum([f(idx) for f in gaussians])
    #     return self.mask


MASKS = (SaltPepperMask, NormalMask, GaussianBlobMask)
