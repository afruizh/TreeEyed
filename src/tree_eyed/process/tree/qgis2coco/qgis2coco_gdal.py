import os
import json
import math
import tempfile
from osgeo import gdal, ogr, osr
import numpy as np
from datetime import date
import cv2 as cv

# Utility functions

def check_raster_gdal(input_file):
    metadata = {}
    ds = gdal.Open(input_file)
    if ds is None:
        raise ValueError(f"Cannot open raster: {input_file}")
    metadata["width"] = ds.RasterXSize
    metadata["height"] = ds.RasterYSize
    metadata["num_bands"] = ds.RasterCount
    metadata["dtype"] = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    gt = ds.GetGeoTransform()
    metadata["spatial_resolution_m"] = abs(gt[1])
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    metadata["crs_units"] = srs.GetAttrValue("UNIT")
    return metadata

# Grid creation using GDAL/OGR

def create_grid_with_raster_reference_gdal(raster_path, cell_w, cell_h, overlap_h=0, overlap_v=0, vector_path=None, save_path=None):
    ds = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + ds.RasterYSize * gt[5]
    maxx = gt[0] + ds.RasterXSize * gt[1]
    maxy = gt[3]
    cell_width = cell_w * abs(gt[1])
    cell_height = cell_h * abs(gt[5])
    overlap_width = overlap_h * abs(gt[1])
    overlap_height = overlap_v * abs(gt[5])
    step_x = cell_width - overlap_width
    step_y = cell_height - overlap_height
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Overlap too large relative to cell size.")
    x_coords = np.arange(minx, maxx, step_x)
    y_coords = np.arange(miny, maxy, step_y)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if save_path:
        if os.path.exists(save_path):
            print(f"File exists: {save_path}")
        else:
            ds_grid = driver.CreateDataSource(save_path)
            srs = osr.SpatialReference()
            srs.ImportFromWkt(ds.GetProjection())
            layer = ds_grid.CreateLayer("grid", srs, ogr.wkbPolygon)
            for x in x_coords:
                for y in y_coords:
                    ring = ogr.Geometry(ogr.wkbLinearRing)
                    ring.AddPoint(x, y)
                    ring.AddPoint(x + cell_width, y)
                    ring.AddPoint(x + cell_width, y + cell_height)
                    ring.AddPoint(x, y + cell_height)
                    ring.AddPoint(x, y)
                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)
                    feat = ogr.Feature(layer.GetLayerDefn())
                    feat.SetGeometry(poly)
                    layer.CreateFeature(feat)
                    feat = None
            ds_grid = None
    return x_coords, y_coords, cell_width, cell_height

# Main class
class QGIS2COCO_GDAL:
    def __init__(self, path_raster, path_vector, category="tree", supercategory="tree", allow_clipped_annotations=True, allow_no_annotations=True, class_column=[], invalid_class=[], preffix='tile_', crs=None, license=None, information=None, contributor=None, license_url=None, output_format=".tif", progress_callback=None, interruption_check=None):
        self.path_raster = path_raster
        self.path_vector = path_vector
        self.category = category
        self.supercategory = supercategory
        self.allow_clipped_annotations = allow_clipped_annotations
        self.allow_no_annotations = allow_no_annotations
        self.class_column = class_column
        self.invalid_class = invalid_class
        self.preffix = preffix
        self.crs = crs
        self.license = license
        self.information = information
        self.contributor = contributor
        self.license_url = license_url
        self.output_format = output_format
        self.progress_callback = progress_callback
        self.interruption_check = interruption_check
        self.raster_ds = gdal.Open(self.path_raster)
        self.vector_ds = ogr.Open(self.path_vector)
        self.grid = None
        self.coco_images = []

    def set_path_output(self, path_output):
        self.path_output = path_output
        self.path_annotations = os.path.join(self.path_output, 'annotations')
        self.path_images = os.path.join(self.path_output, 'images', 'default')

    def create_output_folders(self):
        os.makedirs(self.path_output, exist_ok=True)
        os.makedirs(self.path_annotations, exist_ok=True)
        os.makedirs(self.path_images, exist_ok=True)

    def create_grid(self, cell_w, cell_h, overlap_h=0, overlap_v=0):
        self.x_coords, self.y_coords, self.cell_width, self.cell_height = create_grid_with_raster_reference_gdal(self.path_raster, cell_w, cell_h, overlap_h, overlap_v)

    def extract_tiles(self):
        for i, x in enumerate(self.x_coords):
            for j, y in enumerate(self.y_coords):
                basename = f"{self.preffix}{i:03d}_{j:03d}{self.output_format}"
                filename = os.path.join(self.path_images, basename)
                self.clip_raster(x, y, filename)
                self.coco_images.append({"id": i * len(self.y_coords) + j + 1, "file_name": basename, "width": int(self.cell_width), "height": int(self.cell_height)})

    def clip_raster(self, x, y, filename):
        ds = self.raster_ds
        gt = ds.GetGeoTransform()
        px = int((x - gt[0]) / gt[1])
        py = int((y - gt[3]) / gt[5])
        win_x = px
        win_y = py
        win_w = int(self.cell_width / abs(gt[1]))
        win_h = int(self.cell_height / abs(gt[5]))
        arr = ds.ReadAsArray(win_x, win_y, win_w, win_h)
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(filename, win_w, win_h, ds.RasterCount, ds.GetRasterBand(1).DataType)
        for b in range(ds.RasterCount):
            out_ds.GetRasterBand(b+1).WriteArray(arr[b])
        out_gt = list(gt)
        out_gt[0] = x
        out_gt[3] = y
        out_ds.SetGeoTransform(tuple(out_gt))
        out_ds.SetProjection(ds.GetProjection())
        out_ds.FlushCache()
        out_ds = None

    def extract_annotations(self):
        # Placeholder: implement annotation extraction using OGR
        # This would require spatial intersection between vector features and grid tiles
        pass

    def convert(self, path_output, cell_w, cell_h, overlap_h=0, overlap_v=0):
        self.set_path_output(path_output)
        self.create_output_folders()
        self.create_grid(cell_w, cell_h, overlap_h, overlap_v)
        self.extract_tiles()
        self.extract_annotations()
        # Save COCO dataset as JSON
        file_annotations = os.path.join(self.path_annotations, "instances_default.json")
        with open(file_annotations, "w") as f:
            json.dump({"images": self.coco_images}, f, indent=4)
