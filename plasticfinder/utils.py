import numpy as np
import geopandas as gpd
import pandas as pd
import re
from multiprocessing import Pool
from eolearn.core import EOPatch
from matplotlib import pyplot as plt
from sklearn.covariance import MinCovDet
from timeit import default_timer as timer
from datetime import timedelta
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import transform
from sentinelhub import BBox
from pathlib import Path
from pyproj.crs import CRS
from pyproj import Transformer

from plasticfinder.viz import plot_ndvi_fid_plots, plot_masks_and_vals
from plasticfinder.tasks.detect_plastics import get_feature_names

NTHREAD = 6


def compute_global_distribution_robust(patch_dir):
    pool = Pool(NTHREAD)
    print("Collecting patches...")
    results = pool.map(get_features, list(patch_dir.rglob("feature_*")))
    results = np.vstack(list(filter(lambda x: x is not None, results)))
    print(results.shape)
    if results.shape[0] > 1e6:
        sample = np.random.choice(results.shape[0], int(1e6), replace=False)
        results = results[sample, :]
    print("Fitting robust covariance estimate:")
    start = timer()
    mcd = MinCovDet(store_precision=False).fit(results)
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


def create_outliers_dataset(base_dir):
    pool = Pool(NTHREAD)
    print("Collecting patches...")
    results = pool.starmap(get_geo_df, [(patch, "ROBUST_OUTLIERS") for patch in
                                          list((base_dir / "model_features").rglob("feature_*"))])
    results = list(filter(lambda x: x is not None, results))
    gdf = pd.concat(results, ignore_index=True)
    gdf.to_file(base_dir/"outliers.shp")


def get_features(path, mask_key="FULL_MASK"):
    patch = EOPatch.load(path)
    mask = patch.mask[mask_key].ravel()
    if not np.any(mask):
        return None
    features = patch.data["FEATURES"]
    features = features.reshape((-1, features.shape[-1]))[mask, :]
    return features


def get_geo_df(path, mask_key="FULL_MASK"):
    patch = EOPatch.load(path)
    entries = []
    mask = patch.mask[mask_key]
    if not np.any(mask):
        return None
    features = patch.data["FEATURES"]
    names = get_feature_names()
    for x, y in np.argwhere(mask[0, :, :, 0]):
        polygon = pixel_to_polygon(x, y, patch.bbox)
        entries.append({
            "properties": {name : features[0, x, y, i] for i, name in enumerate(names)},
            "geometry": polygon
        })
    gdf = gpd.GeoDataFrame.from_features(entries, crs=patch.bbox.crs.epsg).to_crs(4326)
    return gdf


def fdi_thresholding(features, p=1e-3):
    fdi = features[:, 1]
    t = np.percentile(fdi, (1 - p) * 100)
    return features[fdi < t, :], t


def plot_ndvis_fdis(patch_dir):
    patch = EOPatch.load(patch_dir)
    if not np.any(patch.mask["FULL_MASK"]):
        return
    fig, ax = plot_ndvi_fid_plots(patch)
    fig.savefig(patch_dir / "ndvi_fdi.png")
    plt.close(fig)


def plot_patch(patch_dir):
    patch = EOPatch.load(patch_dir)
    fig, ax = plot_masks_and_vals(patch)
    fig.savefig(patch_dir / "bands.png")
    plt.close(fig)


def pixel_to_polygon(y, x, bbox, res=10):
    ur = (bbox.min_x + res * (x + 1), bbox.max_y - res * y)
    br = (bbox.min_x + res * (x + 1), bbox.max_y - res * (y + 1))
    bl = (bbox.min_x + res * x, bbox.max_y - res * (y + 1))
    ul = (bbox.min_x + res * x, bbox.max_y - res * y)
    return Polygon([br, ur, ul, bl])


def convert_to_utm(shape, dst_crs, src_crs = CRS("EPSG:4326")):
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
    northern_hemisphere = geo.centroid.coords[0][0] > 0
    if northern_hemisphere:
        utm = CRS(32600 + zone)
    else:
        utm = CRS(32700 + zone)
    geo = convert_to_utm(geo, utm)
    bbox = linestring_to_bbox(geo, utm)
    return bbox, utm
