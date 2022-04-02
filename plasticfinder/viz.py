import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from eolearn.core import LoadTask, FeatureType
from pathlib import Path
from scipy.stats import chi2, norm
from plasticfinder.tasks.detect_plastics import FEATURES, BAND_NAMES, get_features


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

    extent = [patch.bbox.min_x, patch.bbox.max_x, patch.bbox.min_y, patch.bbox.max_y]

    ratio = np.abs(patch.bbox.max_x - patch.bbox.min_x) / np.abs(patch.bbox.max_y - patch.bbox.min_y)
    fig, axs = plt.subplots(3, 5, figsize=(ratio * 10 * 2, 10 * 2))
    axs = axs.flatten()

    axs[0].set_title("True Color")
    patch.plot(feature=(FeatureType.DATA, "TRUE_COLOR"), axes=axs[0], rgb=[0, 1, 2])

    axs[1].set_title("NDVI")
    patch.plot(feature=(FeatureType.DATA, 'NDVI'), axes=axs[1], channels=[0], times=[scene])

    axs[2].set_title("FDI")
    patch.plot(feature=(FeatureType.DATA, 'FDI'), axes=axs[2], channels=[0], times=[scene])

    axs[3].set_title("NDWI")
    patch.plot(feature=(FeatureType.DATA, 'NDWI'), axes=axs[3], channels=[0], times=[scene])

    axs[4].set_title("Data Mask")
    patch.plot(feature=(FeatureType.MASK, 'IS_DATA'), axes=axs[4], channels=[0], times=[scene])

    axs[6].set_title("Water Mask")
    patch.plot(feature=(FeatureType.MASK, 'WATER_MASK'), axes=axs[6], channels=[0], times=[scene])

    axs[6].set_title("Water Mask")
    patch.plot(feature=(FeatureType.MASK, 'WATER_MASK'), axes=axs[6], channels=[0], times=[scene])

    axs[7].set_title("Normed FDI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_FDI'), axes=axs[7], channels=[0], times=[scene])

    axs[8].set_title("Normed NDVI")
    patch.plot(feature=(FeatureType.DATA, 'NORM_NDVI'), axes=axs[8], channels=[0], times=[scene])

    axs[9].set_title("AVG NDVI")
    patch.plot(feature=(FeatureType.DATA, 'MEAN_NDVI'), axes=axs[9], channels=[0], times=[scene])

    axs[10].set_title("AVG FDI")
    patch.plot(feature=(FeatureType.DATA, 'MEAN_FDI'), axes=axs[10], channels=[0], times=[scene])

    axs[11].set_title("Combined mask")
    patch.plot(feature=(FeatureType.MASK, 'FULL_MASK'), axes=axs[11], channels=[0], times=[scene])

    axs[12].set_title("Simple cutoff")
    axs[12].imshow((patch.data['NORM_FDI'][scene, :, :, 0] > 0.005) & (patch.data['NORM_NDVI'][scene, :, :, 0] > 0.1))

    if (points):
        axs[13].set_title('Points')
        axs[13].imshow(patch.data['NORM_FDI'][scene, :, :, 0], extent=extent)
        points.plot(ax=axs[13], markersize=20, color='red')

    if ("SCENE_CLASSIFICATION" in patch.data):
        axs[14].set_title("Labels")
        patch.plot(feature=(FeatureType.DATA, 'SCENE_CLASSIFICATION'), axes=axs[14], channels=[0], times=[scene])

    elif ('CLASSIFICATION' in patch.data):
        classifications = patch.data['CLASSIFICATION'][scene, :, :, 0]

        p_grid = np.array([cols_rgb[val] for val in classifications.flatten()])

        axs[14].set_title("Labels")
        axs[14].imshow(p_grid.reshape(classifications.shape[0], classifications.shape[1], 3))

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
    fig, axs = plt.subplots(3, 3, figsize=(10 * 3, 10 * 3))
    axs = axs.flatten()

    if not np.any(patch.mask["FULL_MASK"]):
        return fig, axs

    levels = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    t_stats_2 = chi2.ppf(1 - levels, 2)

    fdi = FEATURES["fdi"]
    ndvi = FEATURES["ndvi"]
    band_1 = FEATURES["bands"][0]
    band_2 = FEATURES["bands"][1]
    band_3 = FEATURES["bands"][2]
    _, dim1, dim2, _ = patch.data[fdi].shape
    mask = patch.mask["FULL_MASK"].ravel()

    lab_emp = patch.mask["EMPIRICAL_OUTLIERS"].ravel()
    lab_robust = patch.mask["ROBUST_OUTLIERS"].ravel()
    # lab_forest = patch.mask["FOREST_OUTLIERS"].ravel()

    mean_emp = patch.scalar_timeless["EMPIRICAL_MEAN"]
    cov_emp = patch.scalar_timeless["EMPIRICAL_COV"].reshape((5, 5))

    mean_robust = patch.scalar_timeless["ROBUST_MEAN"]
    cov_robust = patch.scalar_timeless["ROBUST_COV"].reshape((5, 5))



    idx = [0, 1]
    axis = 0
    ells = confidence_ellipse(mean_emp[idx], cov_emp[np.ix_(idx, idx)], t_stats_2)
    axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
                      c=lab_emp[mask],
                      cmap="bwr")
    for ell in ells:
        axs[axis].add_artist(ell)
    axs[axis].set_xlabel(ndvi)
    axs[axis].set_ylabel(fdi)

    idx = [0, 1]
    axis = 1
    ells = confidence_ellipse(mean_robust[idx], cov_robust[np.ix_(idx, idx)], t_stats_2)
    axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
                      c=lab_robust[mask],
                      cmap="bwr")
    for ell in ells:
        axs[axis].add_artist(ell)
    axs[axis].set_xlabel(ndvi)
    axs[axis].set_ylabel(fdi)

    # idx = [0, 1]
    # axis = 2
    # axs[axis].scatter(patch.data[ndvi].ravel()[mask], patch.data[fdi].ravel()[mask], s=1.0,
    #                   c=lab_forest[mask],
    #                   cmap="bwr")
    # axs[axis].set_xlabel(ndvi)
    # axs[axis].set_ylabel(fdi)

    # idx = [2, 3]
    # axis = 3
    # band_idx_1 = BAND_NAMES.index(band_1)
    # band_idx_2 = BAND_NAMES.index(band_2)
    # ells = confidence_ellipse(mean_robust[idx], cov_robust[np.ix_(idx, idx)], t_stats_2)
    # axs[axis].scatter(patch.data['NORM_BANDS'][:, :, :, band_idx_1].ravel()[mask],
    #                   patch.data['NORM_BANDS'][:, :, :, band_idx_2].ravel()[mask], s=1.0, c=lab_robust, cmap="bwr")
    # for ell in ells:
    #     axs[axis].add_artist(ell)
    # axs[axis].set_xlabel(ndvi)
    # axs[axis].set_ylabel(fdi)

    range_ = (-0.25, 0.25)
    axis = 4
    x = np.linspace(*range_, 100)
    axs[axis].hist(patch.data[ndvi].ravel()[mask], bins=100, range=range_, density=True)
    axs[axis].plot(x, norm.pdf(x, mean_robust[0], np.sqrt(cov_robust[0, 0])), 'r-')
    axs[axis].plot(x, norm.pdf(x, mean_emp[0], np.sqrt(cov_emp[0, 0])), 'g-')
    axs[axis].set_xlabel(ndvi)
    axs[axis].set_ylabel("count")

    range_ = (-100, 150)
    axis = 5
    x = np.linspace(*range_, 100)
    axs[axis].hist(patch.data[fdi].ravel()[mask], bins=100, range=range_, density=True)
    axs[axis].plot(x, norm.pdf(x, mean_robust[1], np.sqrt(cov_robust[1, 1])), 'r-')
    axs[axis].plot(x, norm.pdf(x, mean_emp[1], np.sqrt(cov_emp[1, 1])), 'g-')
    axs[axis].set_xlabel(fdi)
    axs[axis].set_ylabel("count")

    axis = 3
    axs[axis].set_title("True Color")
    patch.plot(feature=(FeatureType.DATA, "TRUE_COLOR"), axes=axs[axis], rgb=[0, 1, 2])

    axis = 6
    axs[axis].set_title("Empirical Detection")
    axs[axis].imshow(lab_emp.reshape((dim1, dim2)))

    axis = 7
    axs[axis].set_title("Robust Detection")
    axs[axis].imshow(lab_robust.reshape((dim1, dim2)))

    # axis = 8
    # axs[axis].set_title("Forest Detection")
    # axs[axis].imshow(lab_forest.reshape((dim1, dim2)))

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


def plot_classifications(patchDir, features=None):
    ''' Method that will take a given patch plot the results of the model for that patch.
    
        Parameters:
            - patchDir: the directory of the EOPatch to visualize
            - features: Features, could be the training dataset, to overlay on the scatter plots.

        Returns
            Nothing. Will create a file called classifications.png in the EOPatch folder.
    '''
    patch = LoadTask(path=str(patchDir)).execute()
    classifcations = patch.data['CLASSIFICATION'][0, :, :, 0]
    ndvi = patch.data['NDVI'][0, :, :, 0]
    fdi = patch.data['FDI'][0, :, :, 0]
    norm_ndvi = patch.data['NORM_NDVI'][0, :, :, 0]
    norm_fdi = patch.data['NORM_FDI'][0, :, :, 0]

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    axs = axs.flatten()

    fndvi = norm_ndvi.flatten()
    ffdi = norm_fdi.flatten()
    fclassifications = classifcations.flatten()
    fclassifications[(ffdi < 0.007)] = 0

    p_grid = np.array([cols_rgb[val] for val in fclassifications])

    axs[0].set_title("Labels")
    axs[0].imshow(p_grid.reshape(classifcations.shape[0], classifcations.shape[1], 3))

    patch.plot(feature=(FeatureType.DATA, 'NDVI'), axes=axs[1], channels=[0], times=[0])
    axs[1].set_title('NDVI')
    patch.plot(feature=(FeatureType.DATA, 'FDI'), axes=axs[2], channels=[0], times=[0])
    axs[2].set_title('FDI')

    for cat in colors.keys():
        mask = classifcations == cat
        axs[3].scatter(norm_ndvi[mask].flatten(), norm_fdi[mask].flatten(), c=colors[cat], s=0.5, alpha=0.2)
        if (features):
            features.plot.scatter(x='normed_ndvi', y='normed_fdi', ax=axs[3],
                                  color=features.label.apply(lambda l: colors[catMap[l]]))

    axs[4].imshow(norm_ndvi)
    axs[4].set_title('Normed NDVI')

    axs[5].imshow(norm_fdi)
    axs[5].set_title('Normed FDI')

    plt.tight_layout()
    plt.savefig(Path(patchDir) / 'classifications.png')
    plt.close(fig)
