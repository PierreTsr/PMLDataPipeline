from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.core import FeatureType


def get_ndwi_task(band_types='BANDS-S2-L1C'):
    """
        EOTask that calculates the NDWI values
    """

    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11', 'B12']
    ndwi_task = NormalizedDifferenceIndexTask((FeatureType.DATA, band_types), (FeatureType.DATA, 'NDWI'),
                                         [band_names.index('B03'), band_names.index('B08')])
    return ndwi_task
