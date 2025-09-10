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
    def coords2pos(self, minx, miny, maxx, maxy, coord, pixel_w, pixel_h):
        width = abs(maxx - minx)
        height = abs(maxy - miny)
        x = (coord[0] - minx) / width
        y = 1.0 - (coord[1] - miny) / height
        return (x * pixel_w, y * pixel_h)
    
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

    def create_grid(self, cell_w, cell_h, overlap_h=0, overlap_v=0, save_path=None):
        ds = gdal.Open(self.path_raster)
        gt = ds.GetGeoTransform()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
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
        # Create in-memory vector layer for grid
        mem_driver = ogr.GetDriverByName("Memory")
        mem_ds = mem_driver.CreateDataSource("grid_mem")
        grid_layer = mem_ds.CreateLayer("grid", srs, ogr.wkbPolygon)
        grid_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        feature_id = 0
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
                feat = ogr.Feature(grid_layer.GetLayerDefn())
                feat.SetGeometry(poly)
                feat.SetField("id", feature_id)
                grid_layer.CreateFeature(feat)
                feat = None
                feature_id += 1
        self.grid_layer = grid_layer
        self.grid_mem_ds = mem_ds
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.cell_width = cell_width
        self.cell_height = cell_height
        # Optionally save to shapefile
        if save_path:
            shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            if os.path.exists(save_path):
                print(f"File exists: {save_path}")
            else:
                shp_ds = shp_driver.CreateDataSource(save_path)
                shp_layer = shp_ds.CreateLayer("grid", srs, ogr.wkbPolygon)
                shp_layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
                for feat in self.grid_layer:
                    new_feat = ogr.Feature(shp_layer.GetLayerDefn())
                    new_feat.SetGeometry(feat.GetGeometryRef().Clone())
                    new_feat.SetField("id", feat.GetField("id"))
                    shp_layer.CreateFeature(new_feat)
                    new_feat = None
                shp_ds = None

    def extract_tiles(self):
        raster_xsize = self.raster_ds.RasterXSize
        raster_ysize = self.raster_ds.RasterYSize
        nodata = self.raster_ds.GetRasterBand(1).GetNoDataValue()
        if nodata is None:
            nodata = 0
        total = len(self.grid_layer)
        for i, feat in enumerate(self.grid_layer):
            if self.interruption_check and self.interruption_check():
                print("Tile extraction interrupted by user")
                break
            if self.progress_callback:
                progress = (i + 1) / total if total else 1.0
                self.progress_callback({
                    "count": i + 1,
                    "total": total,
                    "progress": progress,
                    "status": "processing",
                    "logs": f"Creating tiles... {i + 1}/{total}"
                })
            geom = feat.GetGeometryRef()
            minx, maxx, miny, maxy = geom.GetEnvelope()[0], geom.GetEnvelope()[1], geom.GetEnvelope()[2], geom.GetEnvelope()[3]
            gt = self.raster_ds.GetGeoTransform()
            px = int(round((minx - gt[0]) / gt[1]))
            py = int(round((maxy - gt[3]) / gt[5]))
            win_w = int(round((maxx - minx) / abs(gt[1])))
            win_h = int(round((maxy - miny) / abs(gt[5])))
            basename = f"{self.preffix}{i:05d}{self.output_format}"
            filename = os.path.join(self.path_images, basename)
            # Calculate intersection with raster bounds
            read_px = max(px, 0)
            read_py = max(py, 0)
            read_w = min(win_w, raster_xsize - read_px)
            read_h = min(win_h, raster_ysize - read_py)
            bands = self.raster_ds.RasterCount
            arr = None
            arr_dtype = np.float32
            if read_w > 0 and read_h > 0:
                arr = self.raster_ds.ReadAsArray(read_px, read_py, read_w, read_h)
                arr_dtype = arr.dtype
            out_arr = np.full((bands, win_h, win_w), nodata, dtype=arr_dtype)
            # Calculate safe assignment sizes
            if arr is not None:
                arr_bands = bands if arr.ndim == 3 else 1
                arr_h = arr.shape[-2] if arr.ndim == 3 else arr.shape[0]
                arr_w = arr.shape[-1] if arr.ndim == 3 else arr.shape[1]
                # Calculate offsets for assignment
                x_off = read_px - px if px < 0 else 0
                y_off = read_py - py if py < 0 else 0
                dest_h = min(arr_h, win_h - y_off)
                dest_w = min(arr_w, win_w - x_off)
                if arr_bands == 1 and arr.ndim == 2:
                    out_arr[0, y_off:y_off+dest_h, x_off:x_off+dest_w] = arr[:dest_h, :dest_w]
                else:
                    out_arr[:arr_bands, y_off:y_off+dest_h, x_off:x_off+dest_w] = arr[:arr_bands, :dest_h, :dest_w]
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(filename, win_w, win_h, bands, self.raster_ds.GetRasterBand(1).DataType)
            for b in range(bands):
                out_ds.GetRasterBand(b+1).WriteArray(out_arr[b])
                out_ds.GetRasterBand(b+1).SetNoDataValue(nodata)
            out_gt = list(gt)
            out_gt[0] = minx
            out_gt[3] = maxy
            out_ds.SetGeoTransform(tuple(out_gt))
            out_ds.SetProjection(self.raster_ds.GetProjection())
            out_ds.FlushCache()
            out_ds = None
            self.coco_images.append({"id": i + 1, "file_name": basename, "width": win_w, "height": win_h})

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
        # Generate COCO-style annotations by intersecting vector features with grid tiles
        annotations = []
        ann_count = 1
        category_id = 1
        vector_layer = self.vector_ds.GetLayer()
        grid_srs = self.grid_layer.GetSpatialRef()
        vector_srs = vector_layer.GetSpatialRef()
        # If CRS do not match, reproject vector layer to grid CRS (default WGS84)
        if not grid_srs.IsSame(vector_srs):
            target_srs = grid_srs if grid_srs else osr.SpatialReference()
            if not grid_srs:
                target_srs.ImportFromEPSG(4326)  # WGS84
            mem_driver = ogr.GetDriverByName("Memory")
            mem_ds = mem_driver.CreateDataSource("vector_mem")
            reprojected_layer = mem_ds.CreateLayer("vector_reprojected", target_srs, geom_type=vector_layer.GetGeomType())
            # Copy fields
            for i in range(vector_layer.GetLayerDefn().GetFieldCount()):
                field_defn = vector_layer.GetLayerDefn().GetFieldDefn(i)
                reprojected_layer.CreateField(field_defn)
            coord_transform = osr.CoordinateTransformation(vector_srs, target_srs)
            for feat in vector_layer:
                geom = feat.GetGeometryRef()
                geom_clone = geom.Clone()
                geom_clone.Transform(coord_transform)
                new_feat = ogr.Feature(reprojected_layer.GetLayerDefn())
                new_feat.SetGeometry(geom_clone)
                for i in range(feat.GetFieldCount()):
                    new_feat.SetField(i, feat.GetField(i))
                reprojected_layer.CreateFeature(new_feat)
                new_feat = None
            vector_layer = reprojected_layer
        total = len(self.coco_images)
        for idx, image_info in enumerate(self.coco_images):
            if self.interruption_check and self.interruption_check():
                print("Annotation extraction interrupted by user")
                break
            if self.progress_callback:
                progress = (idx + 1) / total if total else 1.0
                self.progress_callback({
                    "count": idx + 1,
                    "total": total,
                    "progress": progress,
                    "status": "processing",
                    "logs": f"Extracting annotations... {idx + 1}/{total}"
                })
            image_id = image_info['id']
            # Find corresponding grid feature
            grid_feat = self.grid_layer.GetFeature(image_id - 1)
            grid_geom = grid_feat.GetGeometryRef()
            env = grid_geom.GetEnvelope()
            minx, maxx, miny, maxy = env[0], env[1], env[2], env[3]
            pixel_w = image_info['width']
            pixel_h = image_info['height']
            # Clip vector features to grid tile
            vector_layer.SetSpatialFilter(grid_geom)
            for vfeat in vector_layer:
                vgeom = vfeat.GetGeometryRef()
                print("Vector Geometry:")
                print(vgeom)
                # Intersect geometry with grid tile
                intersection = vgeom.Intersection(grid_geom)
                if intersection is not None and not intersection.IsEmpty():
                    print("Intersection Geometry:")
                    print(intersection)
                    geom_type = intersection.GetGeometryType()
                    # Handle Polygon and MultiPolygon, including 2.5D (Z) types
                    rings = []
                    if geom_type in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                        print("Processing Polygon or Polygon Z")
                        rings = [intersection.GetGeometryRef(0)]
                    elif geom_type in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D):
                        print("Processing MultiPolygon or MultiPolygon Z")
                        for i in range(intersection.GetGeometryCount()):
                            poly = intersection.GetGeometryRef(i)
                            if poly.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
                                rings.append(poly.GetGeometryRef(0))
                    for ring in rings:
                        coords = ring.GetPoints()
                        segmentation = []
                        for pt in coords:
                            x, y = pt[:2]
                            px, py = self.coords2pos(minx, miny, maxx, maxy, (x, y), pixel_w, pixel_h)
                            segmentation.append(px)
                            segmentation.append(py)
                        env_xmin, env_xmax, env_ymin, env_ymax = intersection.GetEnvelope()[0], intersection.GetEnvelope()[1], intersection.GetEnvelope()[2], intersection.GetEnvelope()[3]
                        pxmin, pymin = self.coords2pos(minx, miny, maxx, maxy, (env_xmin, env_ymin), pixel_w, pixel_h)
                        pxmax, pymax = self.coords2pos(minx, miny, maxx, maxy, (env_xmax, env_ymax), pixel_w, pixel_h)
                        w = abs(pxmax - pxmin)
                        h = abs(pymax - pymin)
                        bbox = [pxmin, pymax, w, h]
                        ann = {
                            "id": ann_count,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [segmentation],
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        annotations.append(ann)
                        ann_count += 1
            vector_layer.SetSpatialFilter(None)

        self.coco_annotations = annotations
        # # Add categories
        # categories = [
        #     {"id": 1, "name": self.category, "supercategory": self.supercategory}
        # ]
        # # Save COCO dataset as JSON
        # coco_dataset = {
        #     "images": self.coco_images,
        #     "annotations": annotations,
        #     "categories": categories
        # }
        # file_annotations = os.path.join(self.path_annotations, "instances_default.json")
        # with open(file_annotations, "w") as f:
        #     json.dump(coco_dataset, f, indent=4)

    def convert(self, path_output, cell_w, cell_h, overlap_h=0, overlap_v=0):
        self.set_path_output(path_output)
        self.create_output_folders()
        self.create_grid(cell_w, cell_h, overlap_h, overlap_v)
        self.extract_tiles()
        self.extract_annotations()

        # Create coco dataset
        import pycocotools.coco as coco
        coco_dataset = coco.COCO()

        # Add license and information if any
        #if self.license is not None:
        # Update license information
        coco_dataset.dataset['licenses'] = [
            {
                "name": self.license,
                "id": 1,
                "url": self.license_url
            }
        ]
        if self.information is not None:
            # Update info section
            coco_dataset.dataset['info'] = {
                "contributor": self.contributor,
                "date_created": "",  # Use ISO format
                "description": self.information,
                "url": "",
                "version": "1.0",
                "year": str(date.today().year)
            }

        coco_dataset.dataset["images"] = self.coco_images
        coco_dataset.dataset["annotations"] = self.coco_annotations
        categories = [
            {"id": 1, "name": self.category, "supercategory": self.supercategory},
            # ... add more categories
        ]
        coco_dataset.dataset["categories"] = categories

        file_annotations = os.path.join(self.path_annotations,"instances_default.json" )

        with open(file_annotations, "w") as f:
            json.dump(coco_dataset.dataset, f, indent=4)

