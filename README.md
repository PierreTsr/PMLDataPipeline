# Sentinel-2 processing pipeline to indentify potential floating debris

This is a highly modified version of the DataClinic's pipeline. While preserving the main architecture from the original code, most of the internal components have been changed. Moreover, the whole pipeline now serves a different goal: instead of performing a scene classification of a tile, it produces a dataset of outlier pixels for a given tile and Region of Interest (ROI).

This repo is intended to be used as a submodule for https://github.com/PierreTsr/ML-Climate-Final-Project-Template

## Modifications

The main modifications I brought to this pipeline are:

- Replaced SentinelHub (which is now behind a paywall), by local import of S2 tiles and local functions with S2 tiling files;
- Fixed many components that were broken (because of errors or compatibility issues);
- Upgraded the code to the last version of eolearn's API;
- Made the execution parallelized;
- Added the outlier identification layer;
- Provided a better management of the projections;
- Added new indices;
- Changed the visualizations;

What remains to be done before this fork being usable:
- Properly document the code;
- Provide a Quick-Start guide in this README.md;
