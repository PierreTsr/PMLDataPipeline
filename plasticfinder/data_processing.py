from multiprocessing import Pool

from eolearn.core import EONode, SaveTask, OverwritePermission, EOWorkflow, LoadTask, EOExecutor, FeatureType

from tqdm import tqdm

from plasticfinder.tasks.outlier_detection import GlobalDistribution, OutlierDetection
from plasticfinder.tasks.tools import MergeFeatures
from plasticfinder.utils import NTHREAD, plot_ndvis_fdis, plot_patch, compute_global_distribution_robust, \
    create_outliers_dataset


def post_process_patches(base_dir):
    full_feature_path = base_dir / "full_features"
    partial_feature_path = base_dir / "model_features"

    distrib = compute_global_distribution_robust(partial_feature_path)

    load_node = EONode(LoadTask(path=partial_feature_path, lazy_loading=True), inputs=[])
    add_distrib_node = EONode(GlobalDistribution(), inputs=[load_node])
    outliers_detection_node = EONode(OutlierDetection(), inputs=[add_distrib_node])
    save_node = EONode(SaveTask(path=partial_feature_path,
                                overwrite_permission=OverwritePermission.ADD_ONLY,
                                features=[(FeatureType.SCALAR_TIMELESS, "EMPIRICAL_MEAN"),
                                          (FeatureType.SCALAR_TIMELESS, "EMPIRICAL_COV"),
                                          (FeatureType.MASK, "EMPIRICAL_OUTLIERS"),
                                          (FeatureType.MASK, "ROBUST_OUTLIERS"),
                                          (FeatureType.SCALAR_TIMELESS, "ROBUST_MEAN"),
                                          (FeatureType.SCALAR_TIMELESS, "ROBUST_COV"),
                                          # (FeatureType.MASK, "FOREST_OUTLIERS")
                                          ]),
                       inputs=[outliers_detection_node])
    workflow = EOWorkflow([load_node, add_distrib_node, outliers_detection_node, save_node])

    args = [{
        load_node: {"eopatch_folder": path.name},
        add_distrib_node: {"distrib": distrib},
        save_node: {"eopatch_folder": path.name}
    } for path in list(partial_feature_path.rglob("feature_*"))]

    executor = EOExecutor(workflow, args)
    print("Performing outlier detection")
    executor.run(workers=10)
    print("Creating outlier dataset")
    create_outliers_dataset(base_dir)


def merge_results(base_dir):
    full_feature_path = base_dir / "full_features"
    partial_feature_path = base_dir / "model_features"

    load_partial_node = EONode(LoadTask(path=partial_feature_path), inputs=[])
    save_node = EONode(SaveTask(path=full_feature_path, overwrite_permission=OverwritePermission.ADD_ONLY),
                       inputs=[load_partial_node],
                       features=[])
    workflow = EOWorkflow([load_partial_node, save_node])

    args = [{
        load_partial_node: {"eopatch_folder": path.name},
        save_node: {"eopatch_folder": path.name}
    } for path in list(partial_feature_path.rglob("feature_*"))]

    executor = EOExecutor(workflow, args)
    print("Adding results to the full features")
    executor.run(workers=NTHREAD)


def post_processing_visualizations(base_dir):
    full_feature_path = base_dir / "full_features"
    pool = Pool(14)

    args = list(full_feature_path.rglob("feature_*"))
    print("Plotting standard visualizations:")
    for _ in tqdm(pool.imap_unordered(plot_patch, args), total=len(args)):
        pass
    print("Plotting outlier identification visualizations:")
    for _ in tqdm(pool.imap_unordered(plot_ndvis_fdis, args), total=len(args)):
        pass
