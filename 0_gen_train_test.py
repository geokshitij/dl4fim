#!/usr/bin/env python
# coding: utf-8

import geopandas as gpd
from geotile import GeoTile
import yaml
import os

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Create the output directory if it doesn't exist
os.makedirs('output', exist_ok=True)


shapefile_input = config['paths']['shapefile_input']
filtered_shapefile_output = config['paths']['filtered_shapefile_output']
raster_input = config['paths']['raster_input']
rasterized_output = config['paths']['rasterized_output']
tiles_output = config['paths']['tiles_output']
labels_output = config['paths']['labels_output']

# Load the shapefile
fld = gpd.read_file(shapefile_input)
print(f"Initial shapefile records: {len(fld)}")

# Filter the shapefile to include only zones 'A' and 'AE'
fld_filtered = fld[fld['FLD_ZONE'].isin(['A', 'AE'])]
print(f"Filtered shapefile records: {len(fld_filtered)}")

# Export the filtered shapefile
fld_filtered.to_file(filtered_shapefile_output)
print(f"Filtered shapefile exported to: {filtered_shapefile_output}")

# Load the raster data using GeoTile
gt_train = GeoTile(raster_input)
gt_train.meta

# Generate tiles and save
gt_train.generate_tiles(tiles_output, prefix='train_')

# Rasterize the shapefile and save as .tif
gt_train.rasterization(filtered_shapefile_output, rasterized_output)

# Load the rasterized shapefile and generate tiles
gt_train_y = GeoTile(rasterized_output)
gt_train_y.generate_tiles(labels_output, prefix='train_')

# Preprocess tiles (convert NaN to zero and normalize)
gt_train.generate_tiles(save_tiles=False)
gt_train_y.generate_tiles(save_tiles=False)
gt_train.convert_nan_to_zero()
gt_train.normalize_tiles()

# Save tiles as numpy arrays
x_train_path = config['paths']['x_train']
y_train_path = config['paths']['y_train']
gt_train.save_numpy(x_train_path)
gt_train_y.save_numpy(y_train_path)

# Load numpy arrays and check data properties
import numpy as np
X_train = np.load(x_train_path)
y_train = np.load(y_train_path)
print(X_train.max(), X_train.min(), X_train.dtype, X_train.shape)
print(y_train.shape)
