from multiprocessing import Pool

import numpy as np
from eolearn.core import EONode, SaveTask, OverwritePermission, EOWorkflow, LoadTask, EOExecutor, FeatureType, EOPatch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.outliers_pipeline.plasticfinder.tasks.outlier_detection import GlobalDistribution, OutlierDetection
from src.outliers_pipeline.plasticfinder.utils import compute_global_distribution_robust, create_outliers_dataset
from src.outliers_pipeline.plasticfinder.viz import plot_ndvi_fid_plots, plot_masks_and_vals


def post_process_patches(base_dir, outliers_keys=("LOCAL_OUTLIERS", "GLOBAL_OUTLIERS", "FOREST_OUTLIERS")):

    full_feature_path = base_dir / "full_features"
    partial_feature_path = base_dir / "model_features"

    distrib = compute_global_distribution_robust(partial_feature_path)

    load_node = EONode(LoadTask(path=partial_feature_path, lazy_loading=True), inputs=[])
    add_distrib_node = EONode(GlobalDistribution(), inputs=[load_node])
    outliers_detection_node = EONode(OutlierDetection(), inputs=[add_distrib_node])
    save_full_node = EONode(SaveTask(path=full_feature_path,
                                     overwrite_permission=OverwritePermission.ADD_ONLY,
                                     features=[(FeatureType.SCALAR_TIMELESS, "GLOBAL_MEAN"),
                                               (FeatureType.SCALAR_TIMELESS, "GLOBAL_COV"),
                                               (FeatureType.MASK, "GLOBAL_OUTLIERS"),
                                               (FeatureType.MASK, "LOCAL_OUTLIERS"),
                                               (FeatureType.SCALAR_TIMELESS, "LOCAL_MEAN"),
                                               (FeatureType.SCALAR_TIMELESS, "LOCAL_COV"),
                                               (FeatureType.MASK, "FOREST_OUTLIERS")
                                               ]),
                            inputs=[outliers_detection_node])
    save_partial_node = EONode(SaveTask(path=partial_feature_path,
                                        overwrite_permission=OverwritePermission.ADD_ONLY,
                                        features=[(FeatureType.SCALAR_TIMELESS, "GLOBAL_MEAN"),
                                                  (FeatureType.SCALAR_TIMELESS, "GLOBAL_COV"),
                                                  (FeatureType.MASK, "GLOBAL_OUTLIERS"),
                                                  (FeatureType.MASK, "LOCAL_OUTLIERS"),
                                                  (FeatureType.SCALAR_TIMELESS, "LOCAL_MEAN"),
                                                  (FeatureType.SCALAR_TIMELESS, "LOCAL_COV"),
                                                  (FeatureType.MASK, "FOREST_OUTLIERS")
                                                  ]),
                               inputs=[outliers_detection_node])
    workflow = EOWorkflow([load_node, add_distrib_node, outliers_detection_node, save_full_node, save_partial_node])

    args = [{
        load_node: {"eopatch_folder": path.name},
        add_distrib_node: {"distrib": distrib},
        save_full_node: {"eopatch_folder": path.name},
        save_partial_node: {"eopatch_folder": path.name}
    } for path in list(partial_feature_path.rglob("feature_*"))]

    executor = EOExecutor(workflow, args)
    print("Performing outlier detection")
    executor.run(workers=6)
    print("Creating outlier dataset")
    gdf = create_outliers_dataset(base_dir, dst="outliers.shp", key=outliers_keys)
    return gdf


def pre_processing_visualizations(base_dir):
    full_feature_path = base_dir / "full_features"
    pool = Pool(14)

    args = list(full_feature_path.rglob("feature_*"))
    print("Plotting standard visualizations:")
    for _ in tqdm(pool.imap_unordered(plot_patch, args), total=len(args)):
        pass


def post_processing_visualizations(base_dir):
    full_feature_path = base_dir / "full_features"
    pool = Pool(14)

    args = list(full_feature_path.rglob("feature_*"))
    print("Plotting outlier identification visualizations:")
    for _ in tqdm(pool.imap_unordered(plot_ndvis_fdis, args), total=len(args)):
        pass


def plot_ndvis_fdis(patch_dir):
    patch = EOPatch.load(patch_dir)
    if not np.any(patch.mask["FULL_MASK"]):
        return
    if not "GLOBAL_OUTLIERS" in patch.mask.keys():
        return
    fig, ax = plot_ndvi_fid_plots(patch)
    fig.savefig(patch_dir / "ndvi_fdi.png")
    plt.close(fig)


def plot_patch(patch_dir):
    patch = EOPatch.load(patch_dir)
    fig, ax = plot_masks_and_vals(patch)
    fig.savefig(patch_dir / "bands.png")
    plt.close(fig)
