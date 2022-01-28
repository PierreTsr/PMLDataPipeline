from eolearn.mask import CloudMaskTask


def get_cloud_classifier_task():
    """
        A convenience function that sets up the cloud detection task.

       Configures an instance of the EOTask s2_pixel_cloud_detector and AddCloudMaskTask
    """
    
    cloud_detection_task = CloudMaskTask(processing_resolution='120m',
                                         data_feature="BANDS-S2-L1C",
                                         is_data_feature="IS_DATA",
                                         average_over=16,
                                         dilation_size=8)
    return cloud_detection_task
