from eolearn.mask import CloudMaskTask
from eolearn.core import FeatureType


def get_cloud_classifier_task():
    """
        A convenience function that sets up the cloud detection task.

       Configures an instance of the EOTask s2_pixel_cloud_detector and AddCloudMaskTask
    """
    cloud_detection_task = CloudMaskTask(processing_resolution=120,
                                         data_feature=(FeatureType.DATA, "BANDS-S2-L1C"),
                                         is_data_feature=(FeatureType.MASK, "IS_DATA"),
                                         mono_features=(None, 'CLM_S2C'),
                                         mask_feature=None,
                                         average_over=16,
                                         dilation_size=8)
    return cloud_detection_task
