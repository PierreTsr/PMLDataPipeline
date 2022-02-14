from joblib import load
from eolearn.core import EOTask, FeatureType, AddFeatureTask
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering, KMeans

import numpy as np


class DetectPlastics(EOTask):
    ''' EOTask to apply the plastic detection model.

        Applies the specified model to the EOPatch and 
        adds classifications to the EOPatch.

        This step expects the EOPatch to have a data
        layer called "NORM_NDVI" and "NORM_FDI" which 
        are added in the LocalNorm task. It also expects 
        to have the raw Sentinel bands from either L1C or L2A. 

        Adds the data layer CLASSIFICATION to the EOPatch
        
        Initalization parameters:
            model_file (str): the path to the model file to use

        Run time parameters:
            band_layer(str): the name of the data layer to use for raw Sentinel bands
            band_names(str): the names of each band B01, B02 etc
    '''

    def __init__(self, model_file='model/final_model.joblib'):
        self.model = load(model_file)
        self.add_classification = AddFeatureTask((FeatureType.DATA, "CLASSIFICATION"))

    def execute(self, eopatch, band_layer='BANDS-S2-L1C',
                band_names=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11',
                            'B12']):
        ndvi = eopatch.data['NORM_NDVI']
        fdi = eopatch.data['NORM_FDI']
        bands = eopatch.data[band_layer]

        band_6 = bands[:, :, :, band_names.index('B06')]
        band_7 = bands[:, :, :, band_names.index('B07')]
        band_11 = bands[:, :, :, band_names.index('B11')]

        features = np.dstack([
            ndvi.reshape(ndvi.shape[1], ndvi.shape[2]),
            fdi.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_6.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_7.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_11.reshape(ndvi.shape[1], ndvi.shape[2])
        ])

        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        predicted_labels = self.model.predict(features.reshape((ndvi.shape[1] * ndvi.shape[2], 5)))
        eopatch = self.add_classification(eopatch,  predicted_labels.reshape((1, ndvi.shape[1], ndvi.shape[2], 1)))
        return eopatch


class UnsupervisedPlasticDetector(EOTask):
    def __init__(self):
        # self.model = DBSCAN(eps=5,
        #                     min_samples=5,
        #                     n_jobs=14)
        # self.model = SpectralClustering(n_clusters=5,
        #                                 n_jobs=14)
        self.model = AgglomerativeClustering(n_clusters=3)
        # self.model = KMeans(n_clusters=2)
        self.add_classification = AddFeatureTask((FeatureType.DATA, "CLASSIFICATION"))

    def execute(self, eopatch, band_layer='BANDS-S2-L1C',
                band_names=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B08A', 'B09', 'B10', 'B11',
                            'B12']):
        ndvi = eopatch.data['NORM_NDVI']
        fdi = eopatch.data['NORM_FDI']
        bands = eopatch.data[band_layer]

        band_6 = bands[:, :, :, band_names.index('B06')]
        band_7 = bands[:, :, :, band_names.index('B07')]
        band_11 = bands[:, :, :, band_names.index('B11')]

        features = np.dstack([
            ndvi.reshape(ndvi.shape[1], ndvi.shape[2]),
            fdi.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_6.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_7.reshape(ndvi.shape[1], ndvi.shape[2]),
            band_11.reshape(ndvi.shape[1], ndvi.shape[2])
        ])

        print(features.shape)
        predicted_labels = self.model.fit_predict(features.reshape((ndvi.shape[1] * ndvi.shape[2], 5))) + 1
        print(predicted_labels.min(), predicted_labels.max())
        eopatch = self.add_classification(eopatch,  predicted_labels.reshape((1, ndvi.shape[1], ndvi.shape[2], 1)))
        return eopatch