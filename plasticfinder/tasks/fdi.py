from eolearn.core import EOTask, FeatureType, AddFeatureTask
import numpy as np

DEFAULT_BAND_NAMES = ['B01',
                      'B02',
                      'B03',
                      'B04',
                      'B05',
                      'B06',
                      'B07',
                      'B08',
                      'B08A',
                      'B09',
                      'B10',
                      'B11',
                      'B12']

DEFAULT_BAND_VALUES = [
    443,
    490,
    560,
    665,
    705,
    740,
    783,
    842,
    865,
    940,
    1375,
    1610,
    2190
]

RE = DEFAULT_BAND_NAMES.index("B06")
NIR = DEFAULT_BAND_NAMES.index("B08")
SWIR = DEFAULT_BAND_NAMES.index("B11")

λ_re = DEFAULT_BAND_VALUES[RE]
λ_nir = DEFAULT_BAND_VALUES[NIR]
λ_swir = DEFAULT_BAND_VALUES[SWIR]


class CalcFDI(EOTask):
    """
        EOTask that calculates the floating debris index see https://www.nature.com/articles/s41598-020-62298-z

        Expects the EOPatch to have either Sentinel L1C or L2A bands.

        Will append the data layer "FDI" to the EOPatch

        Run time parameters:
            - band_layer(str): the name of the data layer to use for raw Sentinel bands
            - band_names(str): the names of each band B01, B02 etc
    """

    def __init__(self):
        self.add_feature = AddFeatureTask((FeatureType.DATA, 'FDI'))

    @staticmethod
    def FDI(NIR, RE, SWIR):
        factor = (λ_nir - λ_re) / (λ_swir - λ_re) * 10
        return NIR - (RE + (SWIR - RE) * factor)

    def execute(self,
                eopatch,
                band_layer='BANDS-S2-L1C',
                ):
        bands = eopatch.data[band_layer]

        nir = bands[:, :, :, NIR]
        re = bands[:, :, :, RE]
        swir = bands[:, :, :, SWIR]

        fdi = self.FDI(nir, re, swir).reshape([bands.shape[0], bands.shape[1], bands.shape[2], 1])
        fdi = np.where(np.invert(eopatch.mask["IS_DATA"]), fdi, np.nan)

        eopatch = self.add_feature(eopatch, fdi)
        return eopatch
