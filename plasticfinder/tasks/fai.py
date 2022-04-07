from eolearn.core import EOTask, FeatureType, AddFeatureTask
import numpy as np

from src.outliers_pipeline.plasticfinder.utils import BAND_NAMES, DEFAULT_BAND_VALUES

RE = BAND_NAMES.index("B04")
NIR = BAND_NAMES.index("B08A")
SWIR = BAND_NAMES.index("B11")

λ_re = DEFAULT_BAND_VALUES[RE]
λ_nir = DEFAULT_BAND_VALUES[NIR]
λ_swir = DEFAULT_BAND_VALUES[SWIR]


class CalcFAI(EOTask):
    """
        EOTask that calculates the floating debris index see https://www.nature.com/articles/s41598-020-62298-z

        Expects the EOPatch to have either Sentinel L1C or L2A bands.

        Will append the data layer "FAI" to the EOPatch

        Run time parameters:
            - band_layer(str): the name of the data layer to use for raw Sentinel bands
            - band_names(str): the names of each band B01, B02 etc
    """

    def __init__(self):
        self.add_feature = AddFeatureTask((FeatureType.DATA, 'FAI'))

    @staticmethod
    def FAI(NIR, RE, SWIR):
        factor = (λ_nir - λ_re) / (λ_swir - λ_re)
        return NIR - (RE + (SWIR - RE) * factor)

    def execute(self,
                eopatch,
                band_layer='BANDS-S2-L1C',
                ):
        bands = eopatch.data[band_layer]

        nir = bands[:, :, :, NIR]
        re = bands[:, :, :, RE]
        swir = bands[:, :, :, SWIR]

        fai = self.FAI(nir, re, swir).reshape([bands.shape[0], bands.shape[1], bands.shape[2], 1])
        fai = np.where(eopatch.mask["IS_DATA"], fai, np.nan)

        eopatch = self.add_feature(eopatch, fai)
        return eopatch
