from scipy.ndimage import median_filter, gaussian_filter, generic_filter, minimum_filter
from skimage.filters.rank import mean, median
from eolearn.core import EOTask, FeatureType, AddFeatureTask
import numpy as np


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
        self.add_norm_fdi = AddFeatureTask((FeatureType.DATA, "NORM_FDI"))
        self.add_norm_ndvi = AddFeatureTask((FeatureType.DATA, "NORM_NDVI"))
        self.add_mean_fdi = AddFeatureTask((FeatureType.DATA, "MEAN_FDI"))
        self.add_mean_ndvi = AddFeatureTask((FeatureType.DATA, "MEAN_NDVI"))
        self.add_norm_bands = AddFeatureTask((FeatureType.DATA, "NORM_BANDS"))
        self.add_mean_bands = AddFeatureTask((FeatureType.DATA, "MEAN_BANDS"))
        self.add_norm_fdi = AddFeatureTask((FeatureType.DATA, "NORM_FDI"))
        self.add_norm_ndvi = AddFeatureTask((FeatureType.DATA, "NORM_NDVI"))
        self.add_mean_fdi = AddFeatureTask((FeatureType.DATA, "MEAN_FDI"))
        self.add_mean_ndvi = AddFeatureTask((FeatureType.DATA, "MEAN_NDVI"))
        self.add_norm_bands = AddFeatureTask((FeatureType.DATA, "NORM_BANDS"))
        self.add_mean_bands = AddFeatureTask((FeatureType.DATA, "MEAN_BANDS"))


    @staticmethod
    def normalize(data, mask, method='gaussian', window_size=20):
        mean = np.mean(data, axis=tuple(range(data.ndim - 1)))
        masked_data = np.where(mask, data, mean)
        # TODO: check on that part

        result = np.zeros(shape=masked_data.shape)
        norm_scene = np.zeros(shape=result.shape)

        for time_bin in range(data.shape[0]):
            for freq_bin in range(data.shape[3]):

                scene = masked_data[time_bin, :, :, freq_bin]
                if (method == 'mean'):
                    norm = generic_filter(scene, np.nanmean, size=window_size)
                elif (method == 'median'):
                    norm = median_filter(scene, size=window_size)
                elif (method == 'min'):
                    norm = minimum_filter(scene, size=window_size)
                elif (method == "gaussian"):
                    norm = gaussian_filter(scene, sigma=window_size)
                else:
                    raise Exception("Method needs to be either mean, median or min")
                result[time_bin, :, :, freq_bin] = scene - norm
                norm_scene[time_bin, :, :, freq_bin] = norm

        result = np.where(mask, result, np.nan)
        norm_scene = np.where(mask, norm_scene, np.nan)
        return np.array(result), np.array(norm_scene), np.invert(np.isnan(result))

    def execute(self, eopatch, method='gaussian', window_size=20):
        valid_mask = eopatch.mask['FULL_MASK']
        if np.all(np.invert(valid_mask)):
            eopatch = self.add_norm_fdi(eopatch, np.zeros(eopatch.data['FDI'].shape))
            eopatch = self.add_norm_ndvi(eopatch, np.zeros(eopatch.data['NDVI'].shape))
            eopatch = self.add_mean_fdi(eopatch, np.zeros(eopatch.data['FDI'].shape))
            eopatch = self.add_mean_ndvi(eopatch, np.zeros(eopatch.data['NDVI'].shape))
            eopatch = self.add_norm_bands(eopatch, np.zeros(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch = self.add_mean_bands(eopatch, np.zeros(eopatch.data['BANDS-S2-L1C'].shape))
        else:
            normed_ndvi, m_ndvi, mask = LocalNormalization.normalize(eopatch.data['NDVI'], valid_mask, method=method, window_size=window_size)

            normed_fdi, m_fdi, _ = LocalNormalization.normalize(eopatch.data['FDI'], valid_mask, method=method, window_size=window_size)
            normed_bands, m_bands, _ = LocalNormalization.normalize(eopatch.data['BANDS-S2-L1C'], valid_mask, method=method, window_size=window_size)

            eopatch = self.add_norm_fdi(eopatch, normed_fdi)
            eopatch = self.add_norm_ndvi(eopatch, normed_ndvi)
            eopatch = self.add_mean_fdi(eopatch, m_fdi.reshape(eopatch.data['NDVI'].shape))
            eopatch = self.add_mean_ndvi(eopatch, m_ndvi.reshape(eopatch.data['NDVI'].shape))
            eopatch = self.add_norm_bands(eopatch, normed_bands.reshape(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch = self.add_mean_bands(eopatch, m_bands.reshape(eopatch.data['BANDS-S2-L1C'].shape))
            eopatch.mask["FULL_MASK"] &= mask

        return eopatch


local_norm = LocalNormalization()
