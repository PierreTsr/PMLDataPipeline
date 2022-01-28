from eolearn.io import SentinelHubInputTask
from eolearn.core import FeatureType
from sentinelhub import DataCollection, SHConfig
from datetime import timedelta

config = SHConfig()
resolution = 10
max_cloud_coverage = 0.8
time_difference = timedelta(hours=8)
n_threads = 8

input_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'],
    bands_feature=(FeatureType.DATA, 'BANDS-S2-L1C'),
    additional_data=[(FeatureType.MASK, 'IS_DATA')],
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
