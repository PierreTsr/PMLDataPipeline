from eolearn.features import NormalizedDifferenceIndexTask
from eolearn.core import FeatureType


def get_swi_task(band_types='BANDS-S2-L1C'):
    """
        EOTask that calculates the SWI values

        Expects the EOPatch to have either Sentinel L1C or
        L2A bands.

        Will append the data layer "SWI" to the EOPatch

        Run time parameters:
            - band_types(str): the name of the data layer to use for raw Sentinel bands
    """

    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11', 'B12']
    swi_task = NormalizedDifferenceIndexTask((FeatureType.DATA, band_types), (FeatureType.DATA, 'SWI'),
                                              [band_names.index('B05'), band_names.index('B11')])

    return swi_task
