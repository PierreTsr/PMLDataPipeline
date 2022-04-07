from eolearn.core import EOTask, FeatureType, AddFeatureTask
from scipy.ndimage.filters import gaussian_filter
import numpy as np

from src.outliers_pipeline.plasticfinder.utils import gaussian_nan_filter


class WaterDetector(EOTask):
    """
        Very simple water detector based on SWI threshold.

        Adds the mask layer "WATER_MASK" to the EOPatch.

        Expects the EOPatch to have an "SWI" layer.

        Run time arguments:
            - threshold(float): The cutoff threshold for water.
        
    """

    def __init__(self):
        self.add_feature = AddFeatureTask((FeatureType.MASK, 'WATER_MASK'))

    @staticmethod
    def detect_water(swi, threshold):
        return swi > threshold

    def execute(self, eopatch, sigma=5, threshold=0.2, buffer=5):
        subsampled_swis = np.asarray([gaussian_nan_filter(swi[:,:,0], sigma=sigma) for swi in eopatch.data["SWI"]])
        water_masks = self.detect_water(subsampled_swis, threshold)
        water_masks = propagate_mask(water_masks, buffer)
        eopatch = self.add_feature(eopatch, water_masks.reshape(
            [water_masks.shape[0], water_masks.shape[1], water_masks.shape[2], 1]))
        return eopatch


def propagate_mask(x, window_size):
    y = np.zeros_like(x, dtype=float)
    y[np.invert(x)] = np.nan
    y = gaussian_filter(y, sigma=window_size, truncate=1)
    return np.invert(np.isnan(y))

