import re
from datetime import timedelta, date
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer as timer

import geopandas as gpd
import numpy as np
import pandas as pd
from eolearn.core import EOPatch
from pyproj import Transformer
from pyproj.crs import CRS
from scipy.ndimage.filters import gaussian_filter
from sentinelhub import BBox
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import transform
from sklearn.covariance import MinCovDet

NTHREAD = 6

FEATURES = {
    "fdi": "NORM_FDI",
    "ndvi": "NORM_NDVI",
    # "bands": ["B06", "B07", "B11"]
}

N_FEATURES = 2

BAND_NAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11', 'B12']

INDICES = ["FAI", "FDI", "NDMI", "NDVI", "NDWI", "SWI"]

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


def get_feature_names():
    return [FEATURES["fdi"], FEATURES["ndvi"]]  # + FEATURES["bands"]


def compute_global_distribution_robust(patch_dir):
    pool = Pool(8)
    print("Collecting patches...")
    results = pool.map(get_features, list(patch_dir.rglob("feature_*")))
    results = list(filter(lambda x: x is not None, results))
    if not results:
        return np.zeros((2,)), np.eye(2), 2, 0
    results = np.vstack(results)
    print(results.shape)
    if results.shape[0] > 1e6:
        sample = np.random.choice(results.shape[0], int(1e6), replace=False)
        results = results[sample, :]
    print("Fitting robust covariance estimate:")
    start = timer()
    mcd = MinCovDet(store_precision=False)
    mcd.fit(results)
    end = timer()
    print("FastMCD took :", timedelta(seconds=end - start))
    print(mcd.covariance_)
    mean = mcd.location_
    cov = mcd.covariance_
    return mean, cov, results.shape[1], 0


def compute_global_distribution_empirical(patch_dir):
    pool = Pool(NTHREAD)
    print("Collecting patches...")
    results = pool.map(get_features, list(patch_dir.rglob("feature_*")))
    results = np.vstack(list(filter(lambda x: x is not None, results)))
    print("Fitting empirical covariance:")
    mean = np.mean(results, axis=0)
    cov = np.cov(results, rowvar=False)
    print(cov)
    return mean, cov, results.shape[1], 0


def compute_tile_size(patch_dir):
    pool = Pool(12)
    results = pool.map(get_size, list(patch_dir.rglob("feature_*")))
    return sum(results)


def create_outliers_dataset(base_dir, key="LOCAL_OUTLIERS", dst="outliers.shp", features=None):
    if features is None:
        features = ["NORM_BANDS", "MEAN_BANDS"]
        for idx in INDICES:
            features += ["NORM_" + idx, "MEAN_" + idx]
    pool = Pool(NTHREAD)
    print("Collecting patches...")

    results = pool.starmap(get_geo_df, [(patch, key, features) for patch in
                                        list((base_dir / "full_features").rglob("feature_*"))])
    results = list(filter(lambda x: x is not None, results))
    if not results:
        return None
    gdf = pd.concat(results, ignore_index=True)
    print("Dataset created. Found {n} outliers. Writing to file...".format(n=gdf.shape[0]))
    gdf.to_file(base_dir / dst)
    return gdf


def get_features(path, mask_key="FULL_MASK"):
    patch = EOPatch.load(path)
    mask = patch.mask[mask_key].ravel()
    if not np.any(mask):
        return None
    features = patch.data["FEATURES"]
    features = features.reshape((-1, features.shape[-1]))[mask, :]
    return features


def get_size(path, mask_key="FULL_MASK"):
    patch = EOPatch.load(path, lazy_loading=True)
    return np.sum(patch.mask[mask_key])


def get_geo_df(path, mask_key="FULL_MASK", feature_keys=("FEATURES",)):
    patch = EOPatch.load(path)

    try:
        masks = []
        for key in mask_key:
            masks.append(patch.mask[key])
        mask = np.logical_or.reduce(masks)
    except TypeError:
        mask = patch.mask[mask_key]

    if not np.any(mask):
        return None

    features = []
    names = []
    for key in feature_keys:
        data = patch.data[key]
        features.append(data)
        if data.shape[-1] == 1:
            names.append(key)
        elif key == "BANDS-S2-L1C":
            names += BAND_NAMES
        elif key == "NORM_BANDS":
            names += ["NORM_" + band for band in BAND_NAMES]
        elif key == "MEAN_BANDS":
            names += ["MEAN_" + band for band in BAND_NAMES]
        else:
            names += [key + "_" + str(i) for i in range(data.shape[-1])]
    features = np.concatenate(features, axis=3)

    entries = []
    for x, y in np.argwhere(mask[0, :, :, 0]):
        polygon = pixel_to_polygon(x, y, patch.bbox)
        entry = {name: features[0, x, y, i] for i, name in enumerate(names)}
        entry["patch"] = str(path)
        entry["x"] = x
        entry["y"] = y

        try:
            for key, m in zip(mask_key, masks):
                key_ = key.split("_")[0]
                entry[key_] = m[0, x, y, 0]
        except NameError:
            pass

        entries.append({
            "properties": entry,
            "geometry": polygon
        })
    gdf = gpd.GeoDataFrame.from_features(entries, crs=patch.bbox.crs.epsg)
    return gdf


def fdi_thresholding(features, p=1e-3):
    fdi = features[:, 1]
    t = np.percentile(fdi, (1 - p) * 100)
    return features[fdi < t, :], t


def pixel_to_polygon(y, x, bbox, res=10):
    ur = (bbox.min_x + res * (x + 1), bbox.max_y - res * y)
    br = (bbox.min_x + res * (x + 1), bbox.max_y - res * (y + 1))
    bl = (bbox.min_x + res * x, bbox.max_y - res * (y + 1))
    ul = (bbox.min_x + res * x, bbox.max_y - res * y)
    return Polygon([br, ur, ul, bl])


def convert_to_utm(shape, dst_crs, src_crs=CRS("EPSG:4326")):
    f = Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    new_shape = transform(f, shape)
    return new_shape


def linestring_to_bbox(shape, crs):
    coords = shape.coords
    x = [int(coord[0]) for coord in coords]
    y = [int(coord[1]) for coord in coords]
    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)
    bbox = BBox((x_min, y_min, x_max, y_max), crs)
    return bbox


def get_tile_bounding_box(tile, tiling=Path("data/S2_tiling/Features.shp")):
    tiling = gpd.GeoDataFrame.from_file(tiling)
    tiling.set_index("Name", inplace=True)
    tile_name = re.search("[0-9]{2}[A-Z]{3}", tile)[0]
    zone = int(tile_name[0:2])
    geo = tiling.loc[tile_name, "geometry"]
    if isinstance(geo, MultiLineString):
        raise NotImplementedError("The tile queried overlaps the -180° line in WGS84.")
    utm = None
    northern_hemisphere = geo.centroid.coords[0][1] > 0
    if northern_hemisphere:
        utm = CRS(32600 + zone)
    else:
        utm = CRS(32700 + zone)
    geo = convert_to_utm(geo, utm)
    bbox = linestring_to_bbox(geo, utm)
    return bbox, utm


def get_valid_bbox(bbox_list, roi):
    roi = gpd.GeoSeries.from_file(roi)
    geo = roi.geometry[0]
    crs = roi.crs
    valid = []
    for bbox in bbox_list:
        valid.append(bbox.geometry.intersects(geo))
    return valid


def get_matching_marida_target(tile):
    tile_name = re.search("[0-9]{2}[A-Z]{3}", tile)[0]
    timestamp = re.search("[0-9]{8}T[0-9]{6}", tile)[0]
    timestamp = date(int(timestamp[0:4]), int(timestamp[4:6]), int(timestamp[6:8]))
    target = "S2_" + timestamp.strftime("%-d-%-m-%-y") + "_" + tile_name
    return target


def gaussian_nan_filter(x, sigma, truncate=4.0):
    mask = np.isnan(x)

    u = x.copy()
    u[np.isnan(x)] = 0
    u = gaussian_filter(u, sigma=sigma, truncate=truncate)

    v = 0 * x.copy() + 1
    v[np.isnan(x)] = 0
    v = gaussian_filter(v, sigma=sigma, truncate=truncate)

    result = u / v
    result[mask] = np.nan
    return result
