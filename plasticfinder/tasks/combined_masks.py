from eolearn.core import EOTask, FeatureType, AddFeatureTask
import numpy as np


class CombineMask(EOTask):
    """
        Simple task to combine the various masks in to one

        Run time parameters passed on workflow execution:
        use_water(bool): Include the water mask as part of the full mask. Default is false
    """

    def __init__(self):
        self.add_feature = AddFeatureTask((FeatureType.MASK, "FULL_MASK"))

    def execute(self, eopatch, use_water=True):
        if use_water:
            combined = np.logical_and(eopatch.mask['WATER_MASK'].astype(bool),
                                      eopatch.mask['IS_DATA'].astype(bool))
        else:
            combined = eopatch.mask['IS_DATA'].astype(bool)
        eopatch = self.add_feature(eopatch, combined)
        return eopatch
