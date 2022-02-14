from eolearn.core import EOTask, FeatureType, AddFeatureTask
from scipy.ndimage.filters import gaussian_filter
import numpy as np


class WaterDetector(EOTask):
    """
        Very simple water detector based on NDWI threshold.

        Adds the mask layer "WATER_MASK" to the EOPatch.

        Expects the EOPatch to have an "NDWI" layer.

        Run time arguments:
            - threshold(float): The cutoff threshold for water.
        
    """

    def __init__(self):
        self.add_feature = AddFeatureTask((FeatureType.MASK, 'WATER_MASK'))

    @staticmethod
    def detect_water(ndwi, threshold):
        return ndwi > threshold

    def execute(self, eopatch, sigma=10, threshold=0.2):
        subsampled_ndwis = np.asarray([gaussian_filter(ndwi, sigma=sigma) for ndwi in eopatch.data["NDWI"]])
        water_masks = self.detect_water(subsampled_ndwis, threshold)
        eopatch = self.add_feature(eopatch, water_masks.reshape(
            [water_masks.shape[0], water_masks.shape[1], water_masks.shape[2], 1]))
        return eopatch

