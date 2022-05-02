from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from eolearn.core import LoadTask, FeatureType
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm

from src.outliers_pipeline.plasticfinder.utils import FEATURES, N_FEATURES


def plot_masks_and_vals(patch, points=None, scene=0):
    """
        Method that will take a given patch and plot the various data and mask layers it contains

        Parameters:
            - patch: an EOPatch to visualize
            - points: A list of points to overlay on top of the scene
            - scene: The index of the scene within the EOPatch if there are multiple satellite passes.
        Returns
            (fig,axs) : the output figure and the individual plot axis
    """

    ratio = np.abs(patch.bbox.max_x - patch.bbox.min_x) / np.abs(patch.bbox.max_y - patch.bbox.min_y)
    fig, axs = plt.subplots(2, 7, figsize=(ratio * 8 * 7, 8 * 2))
    axs = axs.flatten()

    axs[0].set_title("True Color")
    patch.plot(feature=(FeatureType.DATA, "TRUE_COLOR"), axes=axs[0], rgb=[0, 1, 2])

    axs[1].set_title("FAI")
    patch.plot(feature=(FeatureType.DATA, 'FAI'), axes=axs[1], channels=[0], times=[scene])

    axs[2].set_title("FDI")
    patch.plot(feature=(FeatureType.DATA, 'FDI'), axes=axs[2], channels=[0], times=[scene])

    axs[3].set_title("NDMI")
    patch.plot(feature=(FeatureType.DATA, 'NDMI'), axes=axs[3], channels=[0], times=[scene])

    axs[4].set_title("NDVI")
    patch.plot(feature=(FeatureType.DATA, 'NDVI'), axes=axs[4], channels=[0], times=[scene])

    axs[5].set_title("NDWI")
    patch.plot(feature=(FeatureType.DATA, 'NDWI'), axes=axs[5], channels=[0], times=[scene])

    axs[6].set_title("SWI")
    patch.plot(feature=(FeatureType.DATA, 'SWI'), axes=axs[6], channels=[0], times=[scene])

    axs[7].set_title("SWIR composite")
    patch.plot(feature=(FeatureType.DATA, "SWIR_COMPOSITE"), axes=axs[7], rgb=[0, 1, 2])

    axs[8].set_title("normalized FAI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_FAI'), axes=axs[8], channels=[0], times=[scene])

    axs[9].set_title("normalized FDI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_FDI'), axes=axs[9], channels=[0], times=[scene])

    axs[10].set_title("normalized NDMI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_NDMI'), axes=axs[10], channels=[0], times=[scene])

    axs[11].set_title("normalized NDVI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_NDVI'), axes=axs[11], channels=[0], times=[scene])

    axs[12].set_title("normalized NDWI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_NDWI'), axes=axs[12], channels=[0], times=[scene])

    axs[13].set_title("normalized SWI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_SWI'), axes=axs[13], channels=[0], times=[scene])

    plt.tight_layout()
    return fig, axs


def plot_ndvi_fid_plots(patch):
    """
        Method that will take a given patch and plot NDVI and FDI relationships.

        Parameters:
            - patch: an EOPatch to visualize
        Returns
            (fig,axs) : the output figure and the individual plot axis
    """
    fig, axs = plt.subplots(3, 3, figsize=(5 * 3, 5 * 3), dpi=200)
    axs = axs.flatten()

    if not np.any(patch.mask["FULL_MASK"]):
        return fig, axs

    levels = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    t_stats_2 = chi2.ppf(1 - levels, 2)

    fdi = FEATURES["fdi"]
    ndvi = FEATURES["ndvi"]
    # band_1 = FEATURES["bands"][0]
    # band_2 = FEATURES["bands"][1]
    # band_3 = FEATURES["bands"][2]
    _, dim1, dim2, _ = patch.data[fdi].shape
    mask = patch.mask["FULL_MASK"].ravel()

    outliers=[]
    if "GLOBAL_OUTLIERS" in patch.mask.keys():
        outliers.append("GLOBAL")
    if "LOCAL_OUTLIERS" in patch.mask.keys():
        outliers.append("LOCAL")
    if "FOREST_OUTLIERS" in patch.mask.keys():
        outliers.append("FOREST")

    if "GLOBAL" in outliers:
        lab_global = patch.mask["GLOBAL_OUTLIERS"].ravel()
        mean_global = patch.scalar_timeless["GLOBAL_MEAN"]
        cov_global = patch.scalar_timeless["GLOBAL_COV"].reshape((N_FEATURES, N_FEATURES))

    if "LOCAL" in outliers:
        lab_local = patch.mask["LOCAL_OUTLIERS"].ravel()
        mean_local = patch.scalar_timeless["LOCAL_MEAN"]
        cov_local = patch.scalar_timeless["LOCAL_COV"].reshape((N_FEATURES, N_FEATURES))

    if "FOREST" in outliers:
        lab_forest = patch.mask["FOREST_OUTLIERS"].ravel()

    if "GLOBAL" in outliers:
        idx = [0, 1]
        axis = 0
        ells = confidence_ellipse(mean_global[idx], cov_global[np.ix_(idx, idx)], t_stats_2)
        axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
                          c=lab_global[mask],
                          alpha=0.2,
                          cmap="bwr")
        for ell in ells:
            axs[axis].add_artist(ell)
        axs[axis].legend(ells, ["p < {x:1.0e}".format(x=x) for x in levels], title="Confidence level:")
        axs[axis].set_title("Global MCD outlier estimation")
        axs[axis].set_xlabel(ndvi)
        axs[axis].set_ylabel(fdi)

    if "LOCAL" in outliers:
        idx = [0, 1]
        axis = 1
        ells = confidence_ellipse(mean_local[idx], cov_local[np.ix_(idx, idx)], t_stats_2)
        axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
                          alpha=0.2,
                          c=lab_local[mask],
                          cmap="bwr")
        for ell in ells:
            axs[axis].add_artist(ell)
        axs[axis].legend(ells, ["p < {x:1.0e}".format(x=x) for x in levels], title="Confidence level:")
        axs[axis].set_title("Local MCD outlier estimation")
        axs[axis].set_xlabel(ndvi)
        axs[axis].set_ylabel(fdi)

    if "FOREST" in outliers:
        idx = [0, 1]
        axis = 2
        axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
                          alpha=0.5,
                          c=lab_forest[mask],
                          cmap="bwr")
        axs[axis].set_title("Forest based outlier estimation")
        axs[axis].set_xlabel(ndvi)
        axs[axis].set_ylabel(fdi)

    # idx = [2, 3]
    # axis = 3
    # band_idx_1 = BAND_NAMES.index(band_1)
    # band_idx_2 = BAND_NAMES.index(band_2)
    # ells = confidence_ellipse(mean_local[idx], cov_local[np.ix_(idx, idx)], t_stats_2)
    # axs[axis].scatter(patch.data['NORM_BANDS'][:, :, :, band_idx_1].ravel()[mask],
    #                   patch.data['NORM_BANDS'][:, :, :, band_idx_2].ravel()[mask], s=1.0, c=lab_local[mask], cmap="bwr")
    # for ell in ells:
    #     axs[axis].add_artist(ell)
    # axs[axis].set_xlabel(ndvi)
    # axs[axis].set_ylabel(fdi)

    axis = 4
    ndvi_data = patch.data[ndvi].ravel()[mask]
    range_ = (np.quantile(ndvi_data, 5e-3), np.quantile(ndvi_data, 1-5e-3))
    x = np.linspace(*range_, 500)
    axs[axis].hist(ndvi_data, bins=100, range=range_, density=True, alpha=0.6, label="Observations")
    if "LOCAL" in outliers:
        axs[axis].plot(x, norm.pdf(x, mean_local[0], np.sqrt(cov_local[0, 0])), 'r-', label="LOCAL")
    if "GLOBAL" in outliers:
        axs[axis].plot(x, norm.pdf(x, mean_global[0], np.sqrt(cov_global[0, 0])), 'g-', label="GLOBAL")
    axs[axis].grid(True, axis="both")
    axs[axis].legend(title="Marginal distributions")
    axs[axis].set_title(ndvi + " distribution")
    axs[axis].set_xlabel(ndvi)
    axs[axis].set_ylabel("Density")

    axis = 5
    fdi_data = patch.data[fdi].ravel()[mask]
    range_ = (np.quantile(fdi_data, 5e-3), np.quantile(fdi_data, 1-5e-3))
    x = np.linspace(*range_, 500)
    axs[axis].hist(fdi_data, bins=100, range=range_, density=True, alpha=0.6, label="Observations")
    if "LOCAL" in outliers:
        axs[axis].plot(x, norm.pdf(x, mean_local[1], np.sqrt(cov_local[1, 1])), 'r-', label="LOCAL")
    if "GLOBAL" in outliers:
        axs[axis].plot(x, norm.pdf(x, mean_global[1], np.sqrt(cov_global[1, 1])), 'g-', label="GLOBAL")
    axs[axis].grid(True, axis="both")
    axs[axis].legend(title="Marginal distributions")
    axs[axis].set_title(fdi + " distribution")
    axs[axis].set_xlabel(fdi)
    axs[axis].set_ylabel("Density")

    axis = 3
    axs[axis].set_title("True Color")
    patch.plot(feature=(FeatureType.DATA, "TRUE_COLOR"), axes=axs[axis], rgb=[0, 1, 2])

    if "GLOBAL" in outliers:
        axis = 6
        axs[axis].set_title("Global Outliers")
        axs[axis].imshow(lab_global.reshape((dim1, dim2)))

    if "LOCAL" in outliers:
        axis = 7
        axs[axis].set_title("Local Outliers")
        axs[axis].imshow(lab_local.reshape((dim1, dim2)))

    if "FOREST" in outliers:
        axis = 8
        axs[axis].set_title("Forest Outliers")
        axs[axis].imshow(lab_forest.reshape((dim1, dim2)))

    plt.tight_layout()
    return fig, axs


# from https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
def confidence_ellipse(center, cov, t_stats_2):
    ells = []
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    for t, color in zip(np.sqrt(t_stats_2), iter(plt.cm.rainbow(np.linspace(0, 1, t_stats_2.size)))):
        ell = Ellipse(xy=center,
                      width=lambda_[0] * 2 * t,
                      height=lambda_[1] * 2 * t,
                      angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_color(color)
        ell.set_facecolor('none')
        ells.append(ell)
    return ells
