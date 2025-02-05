from eolearn.core import EOTask, FeatureType, AddFeatureTask
import numpy as np
import numpy.ma as ma
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from scipy.stats import chi2


class OutlierDetection(EOTask):
    def __init__(self):
        self.mcd = MinCovDet(store_precision=False, assume_centered=True)
        self.forest = IsolationForest(n_estimators=100, contamination=1e-2, n_jobs=1)
        self.add_global = AddFeatureTask((FeatureType.MASK, "GLOBAL_OUTLIERS"))
        self.add_local = AddFeatureTask((FeatureType.MASK, "LOCAL_OUTLIERS"))
        self.add_local_mean = AddFeatureTask((FeatureType.SCALAR_TIMELESS, "LOCAL_MEAN"))
        self.add_local_cov = AddFeatureTask((FeatureType.SCALAR_TIMELESS, "LOCAL_COV"))
        self.add_forest = AddFeatureTask((FeatureType.MASK, "FOREST_OUTLIERS"))

    def get_masked_features(self, eopatch):
        features = eopatch.data["FEATURES"]
        mask = eopatch.mask["FULL_MASK"].ravel()
        if not np.any(mask):
            return None
        k = features.shape[-1]
        features = features.reshape((-1, k))[mask, :]
        return features

    def reshape_features(self, eopatch, data, filler=np.nan):
        mask = eopatch.mask["FULL_MASK"]
        _, h, w, _ = mask.shape
        mask = eopatch.mask["FULL_MASK"].ravel()
        indices = np.argwhere(~mask).ravel() - np.cumsum(~mask)[~mask] + 1
        ext_data = np.insert(data, indices, filler)
        return ext_data.reshape((-1, h, w, 1))

    def global_covariance(self, eopatch, features):
        mean = eopatch.scalar_timeless["GLOBAL_MEAN"]
        k = mean.shape[0]
        cov = eopatch.scalar_timeless["GLOBAL_COV"].reshape((k, k))

        t = np.sum((features - mean) * np.dot(np.linalg.inv(cov), (features - mean).T).T, axis=1)
        outliers = t > chi2.ppf(1 - 1e-7, k)
        outliers = self.reshape_features(eopatch, outliers, False)

        return outliers

    def local_covariance(self, eopatch, features):
        self.mcd.fit(features)
        k = features.shape[-1]
        mean = self.mcd.location_
        cov = self.mcd.covariance_

        t = np.sum((features - mean) * np.dot(np.linalg.inv(cov), (features - mean).T).T, axis=1)
        outliers = t > chi2.ppf(1 - 1e-7, k)
        outliers = self.reshape_features(eopatch, outliers, False)
        return outliers

    def isolation_forest(self, eopatch, features):
        outliers = self.forest.fit_predict(features) == -1
        outliers = self.reshape_features(eopatch, outliers, False)
        return outliers

    def execute(self, eopatch, **kwargs):
        k = eopatch.data["FEATURES"].shape[-1]
        features = self.get_masked_features(eopatch)
        mask = eopatch.mask["FULL_MASK"]

        if "GLOBAL" in kwargs["methods"]:
            if features is None:
                eopatch = self.add_global(eopatch, np.zeros(mask.shape, dtype=np.bool))
            else:
                global_outliers = self.global_covariance(eopatch, features)
                eopatch = self.add_global(eopatch, global_outliers)
        if "LOCAL" in kwargs["methods"]:
            if features is None:
                eopatch = self.add_local(eopatch, np.zeros(mask.shape, dtype=np.bool))
                eopatch = self.add_local_mean(eopatch, np.zeros((k,), dtype=np.bool))
                eopatch = self.add_local_cov(eopatch, np.zeros((k,k), dtype=np.bool).flatten())
            else:
                local_outliers = self.local_covariance(eopatch, features)
                eopatch = self.add_local(eopatch, local_outliers)
                eopatch = self.add_local_mean(eopatch, self.mcd.location_)
                eopatch = self.add_local_cov(eopatch, self.mcd.covariance_.flatten())
        if "FOREST" in kwargs["methods"]:
            if features is None:
                eopatch = self.add_forest(eopatch, np.zeros(mask.shape, dtype=np.bool))
            else:
                forest_outliers = self.isolation_forest(eopatch, features)
                eopatch = self.add_forest(eopatch, forest_outliers)
        return eopatch


class GlobalDistribution(EOTask):
    def __init__(self):
        self.add_mean = AddFeatureTask((FeatureType.SCALAR_TIMELESS, "GLOBAL_MEAN"))
        self.add_cov = AddFeatureTask((FeatureType.SCALAR_TIMELESS, "GLOBAL_COV"))

    def execute(self, eopatch, distrib):
        mean, cov, _, _ = distrib
        eopatch = self.add_mean(eopatch, mean)
        eopatch = self.add_cov(eopatch, cov.flatten())
        return eopatch
