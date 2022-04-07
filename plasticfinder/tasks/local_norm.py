import numpy as np
from eolearn.core import EOTask, FeatureType, AddFeatureTask
from scipy.ndimage import median_filter, generic_filter, minimum_filter

from src.outliers_pipeline.plasticfinder.utils import gaussian_nan_filter, INDICES


class LocalNormalization(EOTask):
    """
        EOPatch that performs a local normalization of FDI and NDVI values

       This task will generate a moving average over the EOPatch of NDVI and FDI
       parameters and subtract these from each pixel to normalize the FDI and NDVI
       relationship.

       The task expects there to be an NDVI and FDI data layer along with a layer
       for Sentinel satellite data.

       Appends the following layers to the EOPatch
            NORM_FDI: Normalized FDI values.
            NORM_NDVI: Normalized NDVI values.
            MEAN_FDI:  The windowed average FDI, used mostly for visualization.
            MEAN_NDVI: The windowed average NDVI, used mostly for visualization.
            NORM_BANDS: Each Sentinel band normalized

       Run time arguments:
            - method: the normalization method, one of min,median,mean
            - window_size: the window over which to perform the normalization in pixels
    """

    def __init__(self):
        self.add_norm_tasks = {}
        self.add_mean_tasks = {}
        for idx in INDICES:
            self.add_norm_tasks[idx] = AddFeatureTask((FeatureType.DATA, "NORM_" + idx))
            self.add_mean_tasks[idx] = AddFeatureTask((FeatureType.DATA, "MEAN_" + idx))
        self.add_norm_bands = AddFeatureTask((FeatureType.DATA, "NORM_BANDS"))
        self.add_mean_bands = AddFeatureTask((FeatureType.DATA, "MEAN_BANDS"))

    @staticmethod
    def normalize(data, mask, method='gaussian', window_size=20):
        result = np.zeros(shape=data.shape)
        norm_scene = np.zeros(shape=result.shape)

        for time_bin in range(data.shape[0]):
            for freq_bin in range(data.shape[3]):

                scene = data[time_bin, :, :, freq_bin]
                if (method == 'mean'):
                    norm = generic_filter(scene, np.nanmean, size=window_size)
                elif (method == 'median'):
                    norm = median_filter(scene, size=window_size)
                elif (method == 'min'):
                    norm = minimum_filter(scene, size=window_size)
                elif (method == "gaussian"):
                    norm = gaussian_nan_filter(scene, sigma=window_size)
                else:
                    raise Exception("Method needs to be either mean, median or min")
                result[time_bin, :, :, freq_bin] = scene - norm
                norm_scene[time_bin, :, :, freq_bin] = norm

        result = np.where(np.invert(mask), result, np.nan)
        norm_scene = np.where(np.invert(mask), norm_scene, np.nan)
        mask = np.all(np.invert(np.isnan(result)), axis=-1, keepdims=True)
        return np.array(result), np.array(norm_scene), mask

    def execute(self, eopatch, method='gaussian', window_size=20):
        invalid_mask = np.invert(eopatch.mask['FULL_MASK'])
        if np.all(invalid_mask):
            for idx in INDICES:
                eopatch = self.add_norm_tasks[idx](eopatch, np.zeros(eopatch.data[idx].shape))
                eopatch = self.add_mean_tasks[idx](eopatch, np.zeros(eopatch.data[idx].shape))

            eopatch = self.add_norm_bands(eopatch, np.zeros(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch = self.add_mean_bands(eopatch, np.zeros(eopatch.data['BANDS-S2-L1C'].shape))
        else:
            for idx in INDICES:
                normed_idx, mean_idx, _ = LocalNormalization.normalize(eopatch.data[idx], invalid_mask,
                                                                       method=method, window_size=window_size)
                eopatch = self.add_norm_tasks[idx](eopatch, normed_idx.reshape(eopatch.data[idx].shape))
                eopatch = self.add_mean_tasks[idx](eopatch, mean_idx.reshape(eopatch.data[idx].shape))

            normed_bands, m_bands, mask = LocalNormalization.normalize(eopatch.data['BANDS-S2-L1C'], invalid_mask,
                                                                       method=method, window_size=window_size)
            eopatch = self.add_norm_bands(eopatch, normed_bands.reshape(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch = self.add_mean_bands(eopatch, m_bands.reshape(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch.mask["FULL_MASK"] &= mask

        return eopatch


local_norm = LocalNormalization()
