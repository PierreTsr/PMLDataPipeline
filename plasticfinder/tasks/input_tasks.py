from eolearn.io import SentinelHubInputTask, ImportFromTiffTask
from eolearn.core import FeatureType, EOTask, EOPatch, AddFeatureTask
from sentinelhub import DataCollection, SHConfig
from datetime import timedelta
import numpy as np


class LocalInputTask(EOTask):
    def __init__(self, folder):
        self.import_task = ImportFromTiffTask(
            feature=(FeatureType.DATA, 'BANDS-S2-L1C'),
            folder=folder,
            image_dtype=np.uint16,
            no_data_value=0,
        )
        self.mask_task = AddFeatureTask(feature=(FeatureType.MASK, "IS_DATA"))

    def execute(self, eopatch=None, **kwargs):
        if eopatch is None and "bbox" in kwargs:
            eopatch = EOPatch(bbox=kwargs["bbox"])
        eopatch = self.import_task(eopatch)
        mask = eopatch.data["BANDS-S2-L1C"] != 0
        mask = np.any(mask, axis=-1, keepdims=True)
        eopatch = self.mask_task(eopatch, mask)
        return eopatch


local_input_task = LocalInputTask(
    "data/Ghana/S2B_MSIL1C_20181031T101139_N0206_R022_T30NZM_20181031T135633_s2resampled.tif"
)

config = SHConfig()
resolution = 10
max_cloud_coverage = 0.8
time_difference = timedelta(hours=8)
n_threads = 8

input_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
    bands_feature=(FeatureType.DATA, 'BANDS-S2-L1C'),
    additional_data=[(FeatureType.MASK, 'dataMask')],
    resolution=resolution,
    maxcc=max_cloud_coverage,
    time_difference=time_difference,
    config=config,
    max_threads=n_threads
)

add_l2a = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L2A,
    bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
    bands_feature=(FeatureType.DATA, 'BANDS-L2-L2A'),
    additional_data=[(FeatureType.MASK, 'SCL')],
    resolution=resolution,
    maxcc=max_cloud_coverage,
    time_difference=time_difference,
    config=config,
    max_threads=n_threads
)
