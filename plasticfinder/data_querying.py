import contextily as cx
import geopandas as gp
import matplotlib.pyplot as plt
from eolearn.core import EONode
from eolearn.core import SaveTask, EOWorkflow, EOExecutor
from eolearn.core.constants import OverwritePermission
from requests import HTTPError
from shapely.geometry import Polygon
from itertools import compress

from src.outliers_pipeline.plasticfinder.tasks.cloud_classifier import get_cloud_classifier_task
from src.outliers_pipeline.plasticfinder.tasks.combined_masks import CombineMask
from src.outliers_pipeline.plasticfinder.tasks.detect_plastics import ExtractFeatures
from src.outliers_pipeline.plasticfinder.tasks.fai import CalcFAI
from src.outliers_pipeline.plasticfinder.tasks.fdi import CalcFDI
from src.outliers_pipeline.plasticfinder.tasks.input_tasks import LocalInputTask
from src.outliers_pipeline.plasticfinder.tasks.local_norm import LocalNormalization
from src.outliers_pipeline.plasticfinder.tasks.ndmi import get_ndmi_task
from src.outliers_pipeline.plasticfinder.tasks.ndvi import get_ndvi_task
from src.outliers_pipeline.plasticfinder.tasks.ndwi import get_ndwi_task
from src.outliers_pipeline.plasticfinder.tasks.swi import get_swi_task
from src.outliers_pipeline.plasticfinder.tasks.water_detection import WaterDetector
from src.outliers_pipeline.plasticfinder.utils import get_tile_bounding_box, get_valid_bbox


def indices_workflow(input_node):
    fai_node = EONode(CalcFAI(), inputs=[input_node])
    fdi_node = EONode(CalcFDI(), inputs=[fai_node])
    ndmi_node = EONode(get_ndmi_task(), inputs=[fdi_node])
    ndvi_node = EONode(get_ndvi_task(), inputs=[ndmi_node])
    ndwi_node = EONode(get_ndwi_task(), inputs=[ndvi_node])
    swi_node = EONode(get_swi_task(), inputs=[ndwi_node])

    return swi_node, [fai_node, fdi_node, ndmi_node, ndvi_node, ndwi_node, swi_node]


def patch_preprocessing_workflow(base_dir, tiles_dir):
    """
        Defines a workflow that will download and process a specific EOPatch.

        The pipline has the following steps:
            - Download data
            - Calculate NDVI
            - Calculate SWI
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

    save_full = SaveTask(path=full_feature_path,
                         overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    save_partial = SaveTask(path=partial_feature_path,
                            overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    input_node = EONode(LocalInputTask(tiles_dir))
    last_idx_node, idx_nodes = indices_workflow(input_node)
    cloud_node = EONode(get_cloud_classifier_task(), inputs=[last_idx_node])
    water_node = EONode(WaterDetector(), inputs=[cloud_node])
    mask_node = EONode(CombineMask(), inputs=[water_node])
    norm_node = EONode(LocalNormalization(), inputs=[mask_node])
    extract_node = EONode(ExtractFeatures(), inputs=[norm_node])
    save_full_node = EONode(save_full, inputs=[norm_node])
    save_partial_node = EONode(save_partial, inputs=[extract_node])

    nodes = [input_node,
             cloud_node,
             water_node,
             mask_node,
             norm_node,
             extract_node,
             save_full_node,
             save_partial_node] + idx_nodes

    nodes_with_args = {
        "input": input_node,
        "water": water_node,
        "mask": mask_node,
        "norm": norm_node,
        "save_full": save_full_node,
        "save_partial": save_partial_node
    }

    workflow = EOWorkflow(nodes)
    return workflow, nodes_with_args


def pre_process_tile(base_dir, tile, tiles_dir, patches=(15, 15), roi=None):
    """
        Defines a workflow that will download and process all EOPatches in a defined region.

        This workflow will download all EOPatches in a given larger region.

        The pipline has the following steps:
            - Download data
            - Calculate NDVI
            - Calculate SWI
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
    full_feature_path = base_dir / tile / "full_features"
    full_feature_path.mkdir(parents=True, exist_ok=True)
    partial_feature_path = base_dir / tile / "model_features"
    partial_feature_path.mkdir(parents=True, exist_ok=True)

    aoi, utm = get_tile_bounding_box(tile)

    bbox_list = aoi.get_partition(num_x=patches[0], num_y=patches[1])
    bbox_list = [bbox for row in bbox_list for bbox in row]
    idxs_x = [x for x in range(patches[0]) for y in range(patches[1])]
    idxs_y = [y for x in range(patches[0]) for y in range(patches[1])]
    idxs = [y + x * patches[1] for x, y in zip(idxs_x, idxs_y)]

    if roi is not None:
        valid = get_valid_bbox(bbox_list, roi)
        bbox_list = list(compress(bbox_list, valid))
        idxs_x = list(compress(idxs_x, valid))
        idxs_y = list(compress(idxs_y, valid))
        idxs = list(compress(idxs, valid))

    # Prepare info of selected EOPatches
    gdf = gp.GeoDataFrame({'index': idxs, 'index_x': idxs_x, 'index_y': idxs_y},
                          crs=utm,
                          geometry=[Polygon(bbox.get_polygon()) for bbox in bbox_list])

    ax = gdf.plot(facecolor='w', edgecolor='r', figsize=(20, 10), alpha=0.3)

    for idx, row in gdf.iterrows():
        geo = row.geometry
        index = row['index']
        ax.text(geo.centroid.x, geo.centroid.y, f'{index}', ha='center', va='center')

    try:
        cx.add_basemap(ax=ax, crs=utm)
    except HTTPError:
        print("Failed to load the matching Open Street Map tile. Ignoring...")
        pass
    plt.savefig(base_dir / tile / ("map" + ".png"))

    total = len(bbox_list)
    workflow, nodes = patch_preprocessing_workflow(base_dir / tile, tiles_dir)
    args = [{
        nodes["input"]: {"bbox": bbox, "tile": tile},
        nodes["water"]: {"sigma": 10, "threshold": 0.15, "buffer": 20},
        nodes["mask"]: {"use_water": True},
        nodes["norm"]: {"method": "gaussian", "window_size": 30},
        nodes["save_partial"]: {"eopatch_folder": f"feature_{idx}"},
        nodes["save_full"]: {"eopatch_folder": f"feature_{idx}"}
    } for idx, bbox in zip(idxs, bbox_list)]
    executor = EOExecutor(workflow, args)
    executor.run(workers=2)
    # The bottleneck here is the hard-drive read speed, so more workers won't speed up the process
