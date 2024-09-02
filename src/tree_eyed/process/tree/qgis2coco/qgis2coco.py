import os
import json
import shapely
import math
import tempfile

import rasterio as rio
import geopandas as gpd
import numpy as np
import pandas as pd

from shapely import geometry
from shapely.geometry import Polygon
from rasterio.mask import mask
from rasterio.enums import Resampling

import pycocotools.coco as coco

"""
To apply scale factor it is only necessary to do it for the raster file on load
"""

class QGIS2COCO():
    """This class is used to convert Raster and Vector layers to a coco dataset"""

    def __init__(self, path_raster, path_vector):

        self.path_raster = path_raster # geotiff data
        self.path_vector = path_vector # a geopandas dataframe
        
        self.path_output = None
        self.path_annotations = None
        self.path_images = None

        self.raster = None
        self.vector = None

        self.gdf = None # geopandas dataframe with bounding box of raster file

        self.coco_images = None

        self.temp_file = None

        #self.crs = "4326"
        self.crs = "6933"

        # Load files
        self.load_files()

        #return
    
    def set_path_output(self, path_output):

        self.path_output = path_output
        self.path_annotations = os.path.join(self.path_output, 'annotations')
        self.path_images = os.path.join(self.path_output, 'images')

    def create_output_folders(self):

        # Create main path
        if not os.path.exists(self.path_output):
            os.makedirs(self.path_output)

        # Create annotations path
        if not os.path.exists(self.path_annotations):
            os.makedirs(self.path_annotations)

        # Create images path
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images)    

        return
    
    def resample_raster(self, scale = 1.0):

        print("resample_raster")

        dataset = self.raster

        print(dataset.profile)

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale),
                int(dataset.width * scale)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        new_meta = self.raster.meta.copy()

        new_meta.update({
            "driver":"Gtiff",
            "height":data.shape[-2], # height starts with shape[1]
            "width":data.shape[-1], # width starts with shape[2]
            "transform":transform
        })

        #temp_dir = tempfile.gettempdir()
        #temp_file_path = os.path.join(temp_dir, "resampled_raster.tif")

        self.temp_file = tempfile.TemporaryFile()
        self.temp_file.open
        print(self.temp_file)

        # When done, make sure to remove the temporary file to avoid clutter:
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)

        with rio.open(self.temp_file.name, "w", **new_meta) as dst:
            dst.write(data) 

        self.raster = rio.open(self.temp_file.name)

    
    def load_files(self):

        self.raster = rio.open(self.path_raster)
        self.vector = gpd.read_file(self.path_vector)
        self.vector = self.vector.to_crs(epsg=self.crs)

        #Create bounding box
        image_geo = self.raster
        bb = [(image_geo.bounds[0], image_geo.bounds[3])
                ,(image_geo.bounds[2], image_geo.bounds[3])
                ,(image_geo.bounds[2], image_geo.bounds[1])
                ,(image_geo.bounds[0], image_geo.bounds[1])]

        self.gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon(bb)])
        #self.gdf = self.gdf.set_crs(epsg="3857")
        self.gdf = self.gdf.set_crs(self.raster.crs)
        self.gdf = self.gdf.to_crs(epsg=self.crs)

        return

    def extract_tiles(self, scale = 1.0):

        #size = 256

        #splitImageIntoCells(self.raster, self.path_images, size)

        coco_images = []

        for i, grid_element in enumerate(self.grid.geometry):
            basename = 'tile_' + str(i) + '.tif'
            filename = os.path.join(self.path_images, basename)

            tile = self.clip_raster(i, filename)
            

            coco_images.append({
                "id": i+1
                , "file_name": basename
                , "width": tile.shape[2]
                , "height": tile.shape[1]
            })   

        self.coco_images = coco_images 

        return
    
    def coords2pos(self, tile_grid, coord, pixel_w, pixel_h):

 
        xmin, ymin, xmax, ymax = tile_grid.total_bounds

        width = abs(xmax - xmin)
        height = abs(ymax - ymin)

        x = (coord[0] - xmin)/width
        y = 1.0 - (coord[1] - ymin)/height

        #res = (round(x*pixel_w, 2),round(y*pixel_h, 2))

        return (x*pixel_w, y*pixel_h)

    def extract_annotations(self):

        coco_dataset = coco.COCO()

        #image_paths = ["D:/local_mydata/ROI/results/coco_dataset/images/tile_5.tif"] #list of paths of the images
        image_paths = []
        for i, polygon in enumerate(self.grid.geometry):
            name = 'tile_' + str(i) + '.tif'
            filename = os.path.join(self.path_images, name)
            image_paths.append(filename)

        # # Add images
        # images = []
        # for i, image_path in enumerate(image_paths):
        #     basename = os.path.basename(image_path)
        #     images.append({
        #         "id": i+1
        #         , "file_name": basename
        #         , "width": width
        #         , "height": height
        #     })
        # coco_dataset.dataset["images"] = images

        coco_dataset.dataset["images"] = self.coco_images


        # Add annotations
        annotations = []
        ann_count = 1

        for item in self.coco_images:

            #print(item)

            image_id = int(item['id'])
            category_id = 1 # TODO:  update how it is established
            tile_grid = self.grid[image_id-1:image_id]
            tile_vector = self.clip_vector(tile_grid)
            

            # Can be variable??
            width = item['width']
            height = item['height']

            for index, polygon in enumerate(tile_vector.geometry):

                # Polygon
                #print(polygon)
                polygon_coords = []
                for coord0 in polygon.exterior.coords:
                    
                    #coord = transform*coord0

                    coord = self.coords2pos(tile_grid, coord0, width, height)
                    
                    x = coord[0]
                    y = coord[1]
                    #print(x)
                    #print(y)
                    polygon_coords.append(x)
                    polygon_coords.append(y)

                # Bounding box
                xmin, ymin, xmax, ymax =  polygon.bounds
                xmin, ymin = self.coords2pos(tile_grid, (xmin,ymin), width, height)
                xmax, ymax = self.coords2pos(tile_grid, (xmax,ymax), width, height)

                w = abs(xmax - xmin)
                h = abs(ymax - ymin)

                bbox = [xmin, ymax, w, h]

                # Create COCO annotation
                ann = {
                    "id":ann_count,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [polygon_coords],
                    #"area": mask.sum().item(),
                    "bbox": bbox,
                    #"score": score,
                }
                ann_count = ann_count+1

                annotations.append(ann)

        
        coco_dataset.dataset["annotations"] = annotations

        # Define categories (if not already defined in your annotations)
        categories = [
            {"id": 1, "name": "tree", "supercategory": "tree"},
            # ... add more categories
        ]
        coco_dataset.dataset["categories"] = categories

        # Save the COCO dataset as a JSON file
        file_annotations = os.path.join(self.path_annotations,"annotations.json" )
        #coco_dataset.save(file_annotations)

        with open(file_annotations, "w") as f:
            json.dump(coco_dataset.dataset, f)

        return
    

    def get_tile_extent(self, rows, scale):

        cols = rows

        # Get original dimensions
        w_raster = self.raster.width
        h_raster = self.raster.height

        width_in_meters = abs(self.raster.bounds.right - self.raster.bounds.left)
        height_in_meters = abs(self.raster.bounds.top - self.raster.bounds.bottom)

        # 
        w_tile = math.floor(w_raster/rows)
        h_tile = math.floor(h_raster/cols)


    def create_grid(self, rows0):

        cols0 = rows0

        xmin, ymin, xmax, ymax = self.gdf.total_bounds

        tile_h = abs(ymax - ymin)/rows0
        tile_w = abs(xmax - xmin)/cols0

        cols = list(np.arange(xmin, xmax, tile_w))
        rows = list(np.arange(ymax, ymin, - tile_h))

        df_grid = pd.DataFrame()

        polygons = []
        count = 0
         
        for y in rows:
            for x in cols:
                       
                polygons.append(Polygon([(x,y), (x+tile_w, y), (x+tile_w, y- tile_h), (x, y- tile_h)]))

                df_item = pd.DataFrame({'id': count}, index=[count])
                df_grid = pd.concat((df_grid, df_item))

                count = count + 1

        #self.grid = gpd.GeoDataFrame(df_item, {'geometry':polygons})
        self.grid = gpd.GeoDataFrame(df_grid, geometry=polygons)

        # Fix index error with "module 'pandas' has no attribute 'Int64Index'"
        self.grid.reset_index(drop=True, inplace=True)
        self.grid.set_index("id", inplace = True)

        #self.grid["row_id"] = self.grid.index + 1
        #self.grid.reset_index(drop=True, inplace=True)
        #self.grid.set_index("row_id", inplace = True)

        self.grid = self.grid.set_crs(epsg=self.crs, allow_override=True)
        #grid.to_file("grid.shp")

    def clip_raster(self, id, filename, scale = 1.0):

        #vector = self.grid.to_crs(self.raster.crs)
        vector = self.grid[id:id+1].to_crs(self.raster.crs)

        raster = self.raster

        if (scale != 1.0):
            # Resample if necessary
            # resample data to target shape
            dataset = raster
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale),
                    int(dataset.width * scale)
                ),
                resampling=Resampling.bilinear
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )

            #

        #tile, tile_transform = mask(self.raster, [vector.geometry[id]], crop=True)
        tile, tile_transform = mask(raster, vector.geometry, crop=True)
        tile_meta = self.raster.meta.copy()

        

        tile_meta.update({
            "driver":"Gtiff",
            "height":tile.shape[1], # height starts with shape[1]
            "width":tile.shape[2], # width starts with shape[2]
            "transform":tile_transform
        })

        with rio.open(filename, 'w', **tile_meta) as dst:
            dst.write(tile)

        return tile

    def clip_vector(self, tile_grid):

        #print(type(self.grid.geometry[id]))

        #tile_vector = self.vector.clip(self.grid.geometry[id])
        tile_vector = gpd.overlay(self.vector, tile_grid)

        # Save for debugging purposes
        # print(tile_vector)

        # tile_vector["row_id"] = tile_vector.index
        # tile_vector.reset_index(drop=True, inplace=True)
        # tile_vector.set_index("row_id", inplace = True)

        # tile_vector.to_file("D:/local_mydata/ROI/sample/tile_" + str(id) + ".shp" )

        return tile_vector
    
    def convert(self, path_output, rows = 1, scale = 1.0):

        # if scale != 1.0:
        #     self.resample_raster()
        
        # # Configure the output folder structure
        self.set_path_output(path_output)
        self.create_output_folders()

        # Create a vector grid for each tile
        self.create_grid(rows)

        # Extract tiles and save
        self.extract_tiles()

        # # Extract annotations
        self.extract_annotations()

        if self.temp_file is not None:
            self.temp_file.close()

        return



        




        

    














