from eolearn.core import EOTask


class MergeFeatures(EOTask):
    def execute(self, *eopatches, **kwargs):
        full, partial = eopatches
        full = full.merge(partial)
        return full
