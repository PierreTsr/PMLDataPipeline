from eolearn.io import SentinelHubInputTask, ImportFromTiffTask
from eolearn.core import FeatureType, EOTask, EOPatch, AddFeatureTask
from sentinelhub import DataCollection, SHConfig, BBox, CRS
from osgeo import gdal, osr
from datetime import timedelta, datetime
from pathlib import Path
from contextlib import redirect_stdout
import os
import re
import numpy as np
import warnings

class NoTileFoundError(Exception):
    pass


def intersect(bbox_a, bbox_b):
    bbox1, bbox2 = bbox_a.transform(CRS.WGS84), bbox_b.transform(CRS.WGS84)
    x_intersect = max(bbox1.min_x, bbox2.min_x) < min(bbox1.max_x, bbox2.max_x)
    y_intersect = max(bbox1.min_y, bbox2.min_y) < min(bbox1.max_y, bbox2.max_y)
    return x_intersect and y_intersect


def get_projcs_code(filename):
    file = gdal.Open(str(filename))
    proj = osr.SpatialReference(wkt=file.GetProjection())
    code = proj.GetAuthorityName("projcs") + ":" + proj.GetAuthorityCode("projcs")
    del file
    return code


def get_bbox(filename):
    file = gdal.Open(str(filename))
    code = get_projcs_code(filename)
    min_x, x_res, x_skew, min_y, y_skew, y_res = file.GetGeoTransform()

    max_x = min_x + (file.RasterXSize * x_res)
    max_y = min_y + (file.RasterYSize * y_res)
    del file
    return BBox((min_x, min_y, max_x, max_y), code)


class LocalInputTask(EOTask):
    def __init__(self, folder):
        self.folder = folder
        self.import_task = ImportFromTiffTask(
            feature=(FeatureType.DATA, 'BANDS-S2-L1C'),
            timestamp_size=1,
            image_dtype=np.float32,
            no_data_value=.0,
        )
        self.mask_task = AddFeatureTask(feature=(FeatureType.MASK, "IS_DATA"))
        self.true_color_task = AddFeatureTask((FeatureType.DATA, "TRUE_COLOR"))
        self.swir_task = AddFeatureTask((FeatureType.DATA, "SWIR_COMPOSITE"))
        self.name_task = AddFeatureTask((FeatureType.META_INFO, "TILE_NAME"))
        self.gain = 1

    @staticmethod
    def get_timestamp(tile):
        match = re.search("[0-9]{8}T[0-9]{6}", tile)[0]
        timestamp = datetime(int(match[0:4]), int(match[4:6]), int(match[6:8]),
                             int(match[9:11]), int(match[11:13]), int(match[13:15]))
        return timestamp

    def load_tile(self, eopatch, filename, tile):
        with open(os.devnull, "w") as std:
            with redirect_stdout(std):
                eopatch = self.import_task(eopatch=eopatch, filename=str(filename))
                eopatch.timestamp = [self.get_timestamp(tile)]
                mask = eopatch.data["BANDS-S2-L1C"] != .0
                mask = np.any(mask, axis=-1, keepdims=True)
                eopatch = self.mask_task(eopatch, mask)
                eopatch = self.true_color_task(
                    eopatch,
                    np.array(eopatch.data["BANDS-S2-L1C"][:, :, :, [3, 2, 1]]*self.gain, dtype=np.float32) / 10000
                )
                eopatch = self.swir_task(
                    eopatch,
                    np.array(eopatch.data["BANDS-S2-L1C"][:, :, :, [12, 8, 3]]*self.gain, dtype=np.float32) / 10000
                )
                eopatch = self.name_task(eopatch, tile)
        return eopatch


    def execute(self, eopatch=None, **kwargs):
        if eopatch is None:
            eopatch = EOPatch()
        if "bbox" in kwargs.keys():
            eopatch.bbox = kwargs["bbox"]
        if eopatch.bbox is None:
            raise Exception("EOPatch needs to have a bounding box to import local files.")

        projcs = eopatch.bbox.crs
        tile = kwargs["tile"]

        s = "*" + tile + "*.tif"
        for path in Path(self.folder).glob(s):
            target = eopatch.bbox
            footprint = get_bbox(path)
            if not intersect(target, footprint):
                continue

            tile_projcs = CRS(get_projcs_code(path))
            if projcs != tile_projcs:
                warnings.warn(f"Warning: a valid tile has been found but isn't in a compatible projection system. Ignoring tile :{path}", ResourceWarning)
                print(projcs, tile_projcs)
                continue

            eopatch = self.load_tile(eopatch, path, tile)
            return eopatch
        raise NoTileFoundError()


local_input_task = LocalInputTask("data/S2_L1C/")

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
