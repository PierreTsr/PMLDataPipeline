"""
   This script downloads all the tiles for a given scene.
   It can be run as follows

   python download_scene.py -scene scenes/ghana.json

   The format of the input specification file is json and should have the following keys

   - minLat: The minimum latitude of the region you are interested in
   - maxLat: The maxium latitude of the region you are interested in
   - minLon: The minimum longitude of the region you are interested in
   - maxLon: The maxium longitude of the region you are interested in
   - timeRange: An array of [start_date, end_date]. All Sentinel passes within this timerange will be downloaded.
   - outputDir: Where the data should be downloaded.

   For an example see the file scenes/gulf.json

"""
import argparse
from pathlib import Path

from plasticfinder.data_processing import post_process_patches, post_processing_visualizations, \
    pre_processing_visualizations
from plasticfinder.data_querying import preprocess_tile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to download a given region and process it in to EOPatches')
    parser.add_argument('--dir', type=str, help='Output directory')
    parser.add_argument('--tile', type=str, help='Name of the S2 L1C tile to process')

    args = parser.parse_args()

    output_dir = Path(args.dir)
    tile = args.tile

    preprocess_tile(output_dir, tile, patches=(15, 15))
    pre_processing_visualizations(output_dir / tile)
    post_process_patches(output_dir / tile)
    post_processing_visualizations(output_dir / tile)
