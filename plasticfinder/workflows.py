from eolearn.core import EOPatch, EONode
from plasticfinder.tasks.combined_masks import CombineMask
from plasticfinder.tasks.cloud_classifier import get_cloud_classifier_task
from plasticfinder.tasks.water_detection import WaterDetector
from plasticfinder.tasks.ndwi import get_ndwi_task
from plasticfinder.tasks.ndvi import get_ndvi_task
from plasticfinder.tasks.fdi import CalcFDI
from plasticfinder.tasks.input_tasks import input_task, local_input_task, NoTileFoundError
from plasticfinder.tasks.local_norm import LocalNormalization
from plasticfinder.tasks.detect_plastics import DetectPlastics, UnsupervisedPlasticDetector, ExtractFeatures
from plasticfinder.class_deffs import catMap, colors

import os
import numpy as np
import numpy.ma as ma
import geopandas as gp
import contextily as cx
import matplotlib.pyplot as plt
from eolearn.core import SaveTask, linearly_connect_tasks, LoadTask, EOWorkflow, OutputTask
from eolearn.core.constants import OverwritePermission
from plasticfinder.viz import plot_ndvi_fid_plots, plot_masks_and_vals
from sentinelhub import UtmZoneSplitter, TileSplitter, DataCollection, BBox, CRS
from shapely.geometry import box, Polygon, shape
from multiprocessing import Pool

NTHREAD = 14


def get_and_process_patch(bbox, timestamps, tile_ids, base_dir, index):
    """
        Defines a workflow that will download and process a specific EOPatch.

        The pipline has the following steps:
            - Download data
            - Calculate NDVI
            - Calculate NDWI
            - Calculate FDI
            - Add cloud mask
            - Add water mask
            - Combine all masks
            - Perform local normalization
            - Save the results.

        Parameters:
            - bounds: The bounding box of the EOPatch we wish to process
            - time_range: An array of [start_time,end_time]. Any satellite pass in that range will be processed.
            - base_dir: the directory to save the patches to
            - index: An index to label this patch

        Returns:
            The EOPatch for this region and time range.
    """
    full_feature_path = base_dir / "full_features"
    partial_feature_path = base_dir / "model_features"

    save_full = SaveTask(path=full_feature_path / f'feature_{index}',
                         overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    save_partial = SaveTask(path=partial_feature_path / f'feature_{index}',
                            overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    extract_feature = ExtractFeatures()

    add_fdi = CalcFDI()
    water_detection = WaterDetector()
    combine_mask = CombineMask()
    local_norm = LocalNormalization()
    output_task = OutputTask("eopatch")

    nodes = linearly_connect_tasks(local_input_task,
                                   get_ndvi_task(),
                                   get_ndwi_task(),
                                   add_fdi,
                                   get_cloud_classifier_task(),
                                   water_detection,
                                   combine_mask,
                                   local_norm,
                                   save_full,
                                   output_task)

    extraction_node = EONode(extract_feature, inputs=[nodes[-3]])
    save_partial_node = EONode(save_partial, inputs=[extraction_node])
    nodes += [extraction_node, save_partial_node]
    workflow = EOWorkflow(nodes)
    feature_result = workflow.execute({
        nodes[0]: {
            'bbox': bbox,
            'timestamps': timestamps,
            'ids': tile_ids
        },
        nodes[-6]: {
            'use_water': True
        },
        nodes[-5]: {
            'method': 'gaussian',
            'window_size': 40,
        }
    })
    patch = feature_result.outputs["eopatch"]
    return patch


def download_region(base_dir, minx, miny, maxx, maxy, time_range, target_tiles=None, patches=(1, 1)):
    """
        Defines a workflow that will download and process all EOPatches in a defined region.

        This workflow will download all EOPatches in a given larger region.

        The pipline has the following steps:
            - Download data
            - Calculate NDVI
            - Calculate NDWI
            - Calculate FDI
            - Add cloud mask
            - Add water mask
            - Combine all masks
            - Perform local normalization
            - Save the results.

        Parameters:
            - base_dir: the directory to save the patches to
            - minx: Min Longitude of the region
            - miny: Min Latitude of the region
            - maxx: Max Longitude of the region
            - maxy: Max Latitude of the region
            - time_range: An array of [start_time,end_time]. Any satellite pass in that range will be processed.
            - target_tiles: A list of tiles to manually include (not used right now)

        Returns:
            Nothing.
    """
    full_feature_path = base_dir / "full_features"
    partial_feature_path = base_dir / "model_features"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(partial_feature_path):
        os.makedirs(partial_feature_path)
    if not os.path.exists(full_feature_path):
        os.makedirs(full_feature_path)

    region = box(minx, miny, maxx, maxy)
    bbox_splitter = TileSplitter(
        [region],
        CRS.WGS84,
        tile_split_shape=patches,
        data_collection=DataCollection.SENTINEL2_L1C,
        time_interval=('2018-10-30', '2018-11-01')
    )

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    # Prepare info of selected EOPatches
    geometry = [Polygon(bbox.transform("EPSG:3857").get_polygon()) for bbox in bbox_list]
    idxs_x = [info['index_x'] for info in info_list]
    idxs_y = [info['index_y'] for info in info_list]
    ids = range(len(info_list))
    tile_ids = [info["ids"] for info in info_list]

    gdf = gp.GeoDataFrame({'index': ids, 'index_x': idxs_x, 'index_y': idxs_y},
                          crs='EPSG:3857',
                          geometry=geometry)

    ax = gdf.plot(facecolor='w', edgecolor='r', figsize=(20, 10), alpha=0.3)

    for idx, row in gdf.iterrows():
        geo = row.geometry
        xindex = row['index_x']
        yindex = row['index_y']
        index = row['index']
        ax.text(geo.centroid.x, geo.centroid.y, f'{index}', ha='center', va='center')

    cx.add_basemap(ax=ax, crs='EPSG:3857')
    plt.savefig(f'{base_dir}/region.png')

    total = len(target_tiles) if target_tiles else len(bbox_list)
    args = [(idx, bbox, info, base_dir, total) for idx, (bbox, info) in enumerate(zip(bbox_list, info_list))]
    # for arg in args:
    #     patch_process(*arg)
    pool = Pool(NTHREAD)
    pool.starmap(patch_process, args)


def plot_ndvis_fdis(patch_dir, distrib):
    patch = EOPatch.load(patch_dir)
    fig, ax = plot_ndvi_fid_plots(patch, *distrib)
    fig.savefig(patch_dir / "ndvi_fdi.png")
    plt.close(fig)


def plot_patch(patch_dir):
    patch = EOPatch.load(patch_dir)
    fig, ax = plot_masks_and_vals(patch)
    fig.savefig(patch_dir / "bands.png")
    plt.close(fig)


def patch_process(index, patch_box, info, base_dir, total):
    timestamps = info["timestamps"]
    tile_ids = info["ids"]
    print("Getting patch ", index, ' of ', total)
    try:
        patch = get_and_process_patch(patch_box, timestamps, tile_ids, base_dir, index)
    except NoTileFoundError:
        print(f"Warning: missing tiles {[id + '.tif' for id in tile_ids]}")


def predict_using_model(patch_dir, model_file, method, window_size):
    """
        Defines a workflow that will perform the prediction step on a given EOPatch.

        For a given EOPatch, use the specified model to apply prediction step.

        Parameters:
            - patch_dir: the directory that contains the patch
            - model_file; the path to the model file.
            - method: The local normalization method, one of 'min', 'median' or 'mean'. This should be the same as the one used to train the model.
            - window_size: The window_size used in the local normalization step. Should be the same as that used to train the model.


        Returns:
            Nothing. Updates the EOPatch on disk.
    """

    path = patch_dir
    if type(path) != str:
        path = str(path)
    save = SaveTask(path=path, overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)
    load_task = LoadTask(path=path)
    local_norm = LocalNormalization()

    detect_plastics = UnsupervisedPlasticDetector()  # DetectPlastics(model_file=model_file)
    nodes = linearly_connect_tasks(load_task, detect_plastics, save)
    workflow = EOWorkflow(nodes)
    workflow.execute({
    })


def extract_targets(patchDir):
    path = patchDir
    if type(path) != str:
        path = str(path)

    patch = LoadTask(path=path).execute()
    box = patch.bbox

    classification = patch.data['CLASSIFICATION'][0, :, :, 0]
    print(classification)
    for coord in np.argwhere(classification == catMap['Debris']):
        print(coord)


def plot_base_visualizations(base_dir):
    full_feature_path = base_dir / "full_features"
    pool = Pool(NTHREAD)

    distrib = compute_global_distribution(base_dir)
    args = [(path, distrib) for path in list(full_feature_path.rglob("feature_*"))]
    pool.starmap(plot_ndvis_fdis, args)
    args = list(full_feature_path.rglob("feature_*"))
    pool.map(plot_patch, args)


def compute_global_distribution(patch_dir):
    pool = Pool(NTHREAD)
    results = pool.map(get_features, list((patch_dir / "model_features").rglob("feature_*")))
    results = np.vstack(list(filter(lambda x: x is not None, results)))
    results, t = fdi_thresholding(results)
    γ = results.mean(axis=0)
    σ = np.cov(results, rowvar=False)
    return γ, σ, results.shape[1], t


def get_features(path):
    patch = EOPatch.load(path)
    mask = patch.mask["FULL_MASK"].ravel()
    if not np.any(mask):
        return None
    features = patch.data["FEATURES"]
    features = features.reshape((-1, features.shape[-1]))[mask, :]
    return features


def fdi_thresholding(features, p=1e-3):
    fdi = features[:, 1]
    t = np.percentile(fdi, (1 - p)*100)
    return features[fdi < t, :], t
