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

from concurrent.futures import ProcessPoolExecutor

from pyproj import CRS



from shapely.geometry import box

from datetime import date
import cv2 as cv

import glob

def check_raster(input_file):

    metadata = {}

    with rio.open(input_file) as src:
        metadata["width"] = src.width  # Image width (pixels)
        metadata["height"] = src.height  # Image height (pixels)

        # spatial resolution
        native_res = src.res  # (pixel_width, pixel_height)
        ds_crs = src.crs if src.crs else (CRS.from_string(crs) if crs else None)
        if ds_crs is None:
            raise ValueError("Raster has no CRS. Provide a valid CRS.")
        
        if ds_crs.is_projected:
            spatial_resolution_m = native_res
        elif ds_crs.is_geographic:
            left, bottom, right, top = src.bounds
            center_lat = (top + bottom) / 2
            xres_deg, yres_deg = native_res
            # Approximate conversion from degrees to meters
            xres_m = xres_deg * 111320 * math.cos(math.radians(center_lat))
            yres_m = yres_deg * 110574
            spatial_resolution_m = (abs(xres_m), abs(yres_m))
        else:
            spatial_resolution_m = native_res
        
        metadata["spatial_resolution_m"] = spatial_resolution_m[0]

        metadata["num_bands"] = src.count        
        
        # Identify data depth
        if np.issubdtype(src.dtypes[0], np.uint8):
            metadata["depth"] = "8-bit"
        elif np.issubdtype(src.dtypes[0], np.uint16):
            metadata["depth"] = "16-bit"
        elif np.issubdtype(src.dtypes[0], np.float32):
            metadata["depth"] = "32-bit float"
        else:
            metadata["depth"] = "Unknown"

        metadata["dtype"] = src.dtypes[0]  # Assuming all bands have the same dtype

        # Check CRS units
        if src.crs:
            if src.crs.is_projected:
                crs_units = src.crs.linear_units  # Often 'm'
            else:
                crs_units = "degrees"
        else:
            crs_units = "unknown"
        metadata["crs_units"] = crs_units

    return metadata

def create_grid_with_raster_reference(raster_path, my_w, my_h, 
                                      overlap_h=0, overlap_v=0, gdf = None, save_path=None):
    """
    Create a grid of rectangular polygons that covers the extent of a GeoDataFrame 
    (or the raster's extent if gdf is None) with optional horizontal and vertical overlap.

    Parameters:
        gdf (GeoDataFrame or None): The input GeoDataFrame whose extent is used.
        raster_path (str): Path to the reference raster file.
        my_w (int or float): The desired cell width in pixel units.
        my_h (int or float): The desired cell height in pixel units.
        overlap_h (int or float): The horizontal overlap between cells in pixel units (default is 0).
        overlap_v (int or float): The vertical overlap between cells in pixel units (default is 0).
        save_path (str or None): Optional file path to save the grid as a shapefile.
    
    Returns:
        grid (GeoDataFrame): A GeoDataFrame containing the grid polygons.
    """
    # Open the raster to get the pixel resolution and extent.
    with rio.open(raster_path) as src:
        # Pixel resolution: transform.a is pixel width, transform.e (usually negative) is pixel height.
        pixel_width = src.transform.a
        pixel_height = -src.transform.e  # take absolute value
        
        # Determine the extent: use gdf if provided, otherwise use raster bounds.
        if gdf is None:
            minx, miny, maxx, maxy = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        else:

            if type(gdf) == gpd.GeoDataFrame: # if gdf is a GeoDataFrame
                minx, miny, maxx, maxy = gdf.total_bounds
            if type(gdf) == str: #if gdf is a path to a shapefile
                gdf = gpd.read_file(gdf)
                minx, miny, maxx, maxy = gdf.total_bounds

    # Convert the desired cell dimensions from pixel units to coordinate units.
    cell_width = my_w * pixel_width
    cell_height = my_h * pixel_height

    # Convert overlaps from pixel units to coordinate units.
    overlap_width = overlap_h * pixel_width
    overlap_height = overlap_v * pixel_height

    # Compute the step sizes for the grid (i.e., how far apart each cell's origin should be).
    # The step size is the cell size minus the desired overlap.
    step_x = cell_width - overlap_width
    step_y = cell_height - overlap_height

    # Validate that the step sizes are positive.
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Overlap is too large relative to the cell size resulting in non-positive step size.")

    # Generate coordinates for grid cells.
    x_coords = np.arange(minx, maxx, step_x)
    y_coords = np.arange(miny, maxy, step_y)

    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            # Create a rectangular polygon for each grid cell.
            cell = box(x, y, x + cell_width, y + cell_height)
            grid_cells.append(cell)

    # Build the GeoDataFrame.
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf.crs if gdf is not None else src.crs)

    # Optionally save the grid to file if a save_path is provided.
    if save_path is not None:
        if not os.path.exists(save_path):
            grid.to_file(save_path)
        else:
            print("File already exists at the specified path:", save_path)

    return grid

def create_tiles_raster(raster_path, gdf, output_dir):

    # Load grid from file or use provided GeoDataFrame.
    if isinstance(gdf, str):
        grid = gpd.read_file(gdf)
    elif isinstance(gdf, gpd.GeoDataFrame):
        grid = gdf
    else:
        raise ValueError("gdf must be a GeoDataFrame or a valid path to a shapefile.")

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)   

    # Open the raster
    with rio.open(raster_path) as raster:

        for id in range(len(grid)):

            output_filename = os.path.join(output_dir, f"tile_{id}.tif")

            if not os.path.exists(output_filename):

                #vector = self.grid.to_crs(self.raster.crs)
                vector = grid[id:id+1].to_crs(raster.crs)

                tile, tile_transform = mask(raster, vector.geometry, crop=True)
                tile_meta = raster.meta.copy()

                tile_meta.update({
                    "driver":"Gtiff",
                    "height":tile.shape[1], # height starts with shape[1]
                    "width":tile.shape[2], # width starts with shape[2]
                    "transform":tile_transform
                })

                with rio.open(output_filename, 'w', **tile_meta) as dst:
                    dst.write(tile)

def process_tile(tile_id, geometry, raster_path, output_dir):
    """
    Process an individual tile: clip the raster to the tile geometry,
    and save the result if it doesn't already exist.
    """
    output_filename = os.path.join(output_dir, f"tile_{tile_id}.tif")
    if os.path.exists(output_filename):
        return  # Skip if file already exists

    # Open the raster within the worker process
    with rio.open(raster_path) as raster:
        tile, tile_transform = mask(raster, [geometry], crop=True)
        tile_meta = raster.meta.copy()
        tile_meta.update({
            "driver": "GTiff",
            "height": tile.shape[1],
            "width": tile.shape[2],
            "transform": tile_transform
        })

    with rio.open(output_filename, 'w', **tile_meta) as dst:
        dst.write(tile)
    return output_filename

def create_tiles_raster_parallel(raster_path, gdf, output_dir, num_workers=4):

    # Load grid from file or use provided GeoDataFrame.
    if isinstance(gdf, str):
        grid = gpd.read_file(gdf)
    elif isinstance(gdf, gpd.GeoDataFrame):
        grid = gdf
    else:
        raise ValueError("gdf must be a GeoDataFrame or a valid path to a shapefile.")

    # Ensure output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Open the raster once to obtain its CRS.
    with rio.open(raster_path) as raster:
        raster_crs = raster.crs

    # Reproject the entire grid to the raster's CRS if necessary.
    if grid.crs != raster_crs:
        grid = grid.to_crs(raster_crs)

    # Process each tile in parallel.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for tile_id, row in grid.iterrows():
            geom = row.geometry
            futures.append(
                executor.submit(process_tile, tile_id, geom, raster_path, output_dir)
            )
        # Optionally, wait for all tasks to finish.
        for future in futures:
            future.result()

    print("Tile creation complete.")

    
# def create_tiles_raster(raster_path, gdf, output_dir):

#     if type(gdf) == gpd.GeoDataFrame: # if gdf is a GeoDataFrame
#         grid = gdf
#     if type(gdf) == str: #if gdf is a path to a shapefile
#         grid = gpd.read_file(gdf)        

#     # Open the raster
#     with rio.open(raster_path) as raster:

#         tiles, tiles_transform = mask(raster, grid.geometry, crop=True)
#         tile_meta = raster.meta.copy()

#         for id, tile in enumerate(tiles):

#             output_filename = os.path.join(output_dir, f"tile_{id}.tif")

#             if not os.path.exists(output_filename):

#                 print(tile.shape)

#                 new_tile_meta = tile_meta.copy()                

#                 new_tile_meta.update({
#                     "driver":"Gtiff",
#                     "height":tile.shape[1], # height starts with shape[1]
#                     "width":tile.shape[2], # width starts with shape[2]
#                     "transform":tiles_transform[id]
#                 })

#                 with rio.open(output_filename, 'w', **new_tile_meta) as dst:
#                     dst.write(tile)

class TILER():

    def __init__(self, path_raster, path_vector, category="tree", supercategory="tree"
            ,allow_clipped_annotations = True, allow_no_annotations=True, class_column = [], invalid_class=[]
            , preffix = 'tile_', crs = "6933", license = None, information = None, contributor = None, license_url = None
            , output_format = ".tif"
            , progress_callback = None, interruption_check = None
        ):

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
        #self.crs = "6933" #units in meters
        self.crs = crs        

        self.supercategory = supercategory
        self.category = category
        self.allow_clipped_annotations = allow_clipped_annotations
        self.allow_no_annotations = allow_no_annotations

        self.class_column = class_column
        self.invalid_class = invalid_class

        self.preffix = preffix

        self.license = license
        self.information = information
        self.contributor = contributor
        self.license_url = license_url

        self.output_format = output_format

        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

        # Load files
        self.load_files()

    def load_files(self):

        try:

            self.raster = rio.open(self.path_raster)

            #Create bounding box
            image_geo = self.raster
            bb = [(image_geo.bounds[0], image_geo.bounds[3])
                    ,(image_geo.bounds[2], image_geo.bounds[3])
                    ,(image_geo.bounds[2], image_geo.bounds[1])
                    ,(image_geo.bounds[0], image_geo.bounds[1])]
            
            safe_crs = CRS.from_user_input(self.raster.crs) 

            # self.gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon(bb)], crs=safe_crs)
            # #self.gdf = self.set_crs(epsg="3857")
            # self.gdf = self.gdf.set_crs(self.raster.crs)
            # if self.gdf.crs.to_epsg() != int(self.crs):
            #     self.gdf = self.gdf.to_crs(epsg=int(self.crs))

        except Exception as e:
            # IMPORTANT: Log or report inside the task
            print(f"[load_files] Exception occurred: {e}")
            import traceback
            traceback.print_exc()

        

    def create_grid(self, rows0, w_overlap=0, h_overlap=0):

        cols0 = rows0

        xmin, ymin, xmax, ymax = self.gdf.total_bounds

        Dh = abs(ymax - ymin)
        # oh = Dh*h_overlap
        # tile_h = (Dh - oh)/rows0 + oh

        tile_h = Dh/(rows0-rows0*h_overlap+h_overlap)
        oh = tile_h*h_overlap

        Dw = abs(xmax - xmin)
        # ow = Dw*w_overlap
        # tile_w = (Dw - ow)/cols0 + ow

        tile_w = Dw/(cols0-cols0*w_overlap+w_overlap)
        ow = tile_w*w_overlap


        # # Use only the value for the max 
        # if Dw > Dh and Dh/Dw > 0.7:
        #     tile_h = tile_w
        #     oh = ow
        # elif Dh > Dw and Dw/Dh > 0.7:
        #     tile_w = tile_h
        #     ow = oh

        # # Use only the value for the max 
        if Dw > Dh:
            tile_h = tile_w
            oh = ow
        elif Dh > Dw:
            tile_w = tile_h
            ow = oh

        #tile_h = abs(ymax - ymin)/rows0
        #tile_w = abs(xmax - xmin)/cols0

        #cols = list(np.arange(xmin, xmax, tile_w))
        #rows = list(np.arange(ymax, ymin, - tile_h))

        #print("TILE SIZE")
        #print(tile_h)
        #print(tile_w)

        cols = list(np.arange(xmin, xmax-(ow), tile_w - ow))
        rows = list(np.arange(ymax, ymin-(-(oh)), - (tile_h-oh)))

        #print(cols)
        #print(rows)

        df_grid = pd.DataFrame()

        polygons = []
        count = 0
         
        for y in rows:
            for x in cols:

                poly = Polygon([(x,y), (x+tile_w, y), (x+tile_w, y- tile_h), (x, y- tile_h)])

                polygons.append(poly)

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

    def create_grid_px(self, tile_in_px, overlap=0):
        """
        Create a grid of tiles where each tile is tile_in_px x tile_in_px pixels in size,
        using the actual raster pixel size and CRS. Overlap is also in pixels.

        Parameters:
            tile_in_px (int): Tile size in pixels (width and height).
            w_overlap (float): Horizontal overlap in pixels.
            h_overlap (float): Vertical overlap in pixels.
        """

        from rasterio.windows import Window, bounds
        from shapely.geometry import box
        import geopandas as gpd


        polys, records = [], []

        tile_w = tile_in_px
        tile_h = tile_in_px
        boundless = True      

        with rio.open(self.path_raster) as src:
            ncols, nrows = src.width, src.height
            stride_x = tile_w - overlap                     # move by **core** size
            stride_y = tile_h - overlap
            crs = src.crs
            raster_window = Window(0, 0, ncols, nrows)

            tile_id = 0
            for row_off in range(0, nrows, stride_y):
                for col_off in range(0, ncols, stride_x):
                    # upper-left corner of the core area
                    core = Window(col_off, row_off,
                                  tile_w, tile_h)
                                #min(tile_w, ncols - col_off),
                                #min(tile_h, nrows - row_off))

                    # # expand core by the margin (may poke outside raster)
                    # full = Window(core.col_off - overlap,
                    #             core.row_off - overlap,
                    #             core.width  + 2 * overlap,
                    #             core.height + 2 * overlap)

                    full = core

                    if not boundless:
                        full = full.intersection(raster_window)

                    # polygon in dataset CRS
                    left, bottom, right, top = bounds(full, src.transform)
                    polys.append(box(left, bottom, right, top))

                    records.append({
                        "id": tile_id,
                        "tile":   tile_id,
                        "row_off": int(full.row_off),
                        "col_off": int(full.col_off),
                        "width":  int(full.width),
                        "height": int(full.height)
                    })
                    tile_id += 1

        print(crs)
        self.grid = gpd.GeoDataFrame(records, geometry=polys, crs=crs)
        

        #self.grid = gpd.GeoDataFrame(df_grid, geometry=polygons)
        self.grid.reset_index(drop=True, inplace=True)
        self.grid.set_index("id", inplace=True)
        #self.grid = self.grid.set_crs(epsg=self.crs, allow_override=True)

        #save grid to file
        #self.grid.to_file(r"D:\local_mydata\tree_eye_tests\temporal\grid.shp")


    def extract_tiles(self, scale = 1.0, progress_callback = None, interruption_check = None):

            #size = 256

            #splitImageIntoCells(self.raster, self.path_images, size)

            coco_images = []

            for i, grid_element in enumerate(self.grid.geometry):
                basename = f"{self.preffix}{i:05d}{self.output_format}"
                filename = os.path.join(self.path_images, basename)

                if os.path.exists(filename):
                    print(f"File already exists {filename}")
                    continue


                tile, w, h = self.clip_raster(i, filename)
                

                coco_images.append({
                    "id": i+1
                    , "file_name": basename
                    , "width": w
                    , "height": h
                })

                if self.progress_callback is not None:
                    count = i
                    total = len(self.grid.geometry)
                    progress = count/total
                    status = "processing"
                    logs = "Tiling..."
                    info = {
                        "count": count
                        , "total": total
                        , "progress": progress
                        , "status": status
                        , "logs": logs
                    }
                    self.progress_callback(info)

                if self.interruption_check is not None:
                    if self.interruption_check():
                        break  

            self.coco_images = coco_images 

            return

    def clip_raster(self, id, filename, scale = 1.0, boundless = True):
        """Clip the raster to the geometry of the grid element with the given id and save it to filename.

        """

        from rasterio.features import geometry_mask

        #vector = self.grid.to_crs(self.raster.crs)
        #vector = self.grid[id:id+1].to_crs(self.raster.crs)
        safe_crs = CRS.from_user_input(self.raster.crs) 
        vector = self.grid[id:id+1].to_crs(safe_crs)

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

        #tile, tile_transform = mask(self.raster, [vector.geometry[id]], crop=True)
        #tile, tile_transform = mask(raster, vector.geometry, crop=True, filled = True)

        if boundless:

            # Compute the window covering the vector geometry
            bounds = vector.geometry.total_bounds
            window = rio.windows.from_bounds(*bounds, transform=raster.transform)
            window = window.round_offsets().round_lengths()

            # Read the raster data for the window
            data = raster.read(window=window, boundless=True, fill_value=raster.nodata if raster.nodata is not None else 0)

            # Create a mask for the geometry in the window
            mask_arr = geometry_mask(
                [geom for geom in vector.geometry],
                out_shape=(window.height, window.width),
                transform=raster.window_transform(window),
                invert=True,
                all_touched=True
            )

            # Apply the mask to each band
            tile = np.where(mask_arr, data, raster.nodata if raster.nodata is not None else 0)
            tile_transform = raster.window_transform(window)

        else:
            tile, tile_transform = mask(raster, vector.geometry, crop=True)


        width = tile.shape[2]
        height = tile.shape[1]

        

        if "tif" in self.output_format:

            tile_meta = self.raster.meta.copy()

            tile_meta.update({
                "driver":"Gtiff",
                "height":height, # height starts with shape[1]
                "width":width, # width starts with shape[2]
                "transform":tile_transform
            })
            #print("TILE", tile.shape[1], tile.shape[2])
            with rio.open(filename, 'w', **tile_meta) as dst:
                dst.write(tile)

        else:
            
            tile = np.transpose(tile, (1,2,0))
            # print(type(tile))
            # print(tile.shape)
            # print(filename)
            tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)
            cv.imwrite(filename, tile)

        return tile, width, height



"""
To apply scale factor it is only necessary to do it for the raster file on load
"""


class QGIS2COCO():
    """This class is used to convert Raster and Vector layers to a coco dataset"""

    def __init__(self, path_raster, path_vector, category="tree", supercategory="tree"
            ,allow_clipped_annotations = True, allow_no_annotations=True, class_column = [], invalid_class=[]
            , preffix = 'tile_', crs = "6933", license = None, information = None, contributor = None, license_url = None
            , output_format = ".tif", progress_callback = None, interruption_check = None
        ):

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
        #self.crs = "6933" #units in meters
        self.crs = crs

        # Load files
        self.load_files()

        self.supercategory = supercategory
        self.category = category
        self.allow_clipped_annotations = allow_clipped_annotations
        self.allow_no_annotations = allow_no_annotations

        self.class_column = class_column
        self.invalid_class = invalid_class

        self.preffix = preffix

        self.license = license
        self.information = information
        self.contributor = contributor
        self.license_url = license_url

        self.output_format = output_format

        self.progress_callback = progress_callback
        self.interruption_check = interruption_check

        #return
    
    def set_path_output(self, path_output):

        self.path_output = path_output
        self.path_annotations = os.path.join(self.path_output, 'annotations')
        self.path_images = os.path.join(self.path_output, 'images', 'default')

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
            basename = self.preffix + str(i) + self.output_format
            filename = os.path.join(self.path_images, basename)

            tile, w, h = self.clip_raster(i, filename)
            

            coco_images.append({
                "id": i+1
                , "file_name": basename
                , "width": w
                , "height": h
            })

            if self.progress_callback is not None:
                count = i
                total = len(self.grid.geometry)
                progress = count/total*0.4
                status = "processing"
                logs = "Tiling..."
                info = {
                    "count": count
                    , "total": total
                    , "progress": progress
                    , "status": status
                    , "logs": logs
                }
                self.progress_callback(info)

            if self.interruption_check is not None:
                if self.interruption_check():
                    break


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



        # #image_paths = ["D:/local_mydata/ROI/results/coco_dataset/images/tile_5.tif"] #list of paths of the images
        # image_paths = []
        # for i, polygon in enumerate(self.grid.geometry):
        #     name = self.preffix + str(i) + self.output_format
        #     filename = os.path.join(self.path_images, name)
        #     image_paths.append(filename)

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


        width = 0
        height = 0
        image_index = 0


        for index, item in enumerate(self.coco_images):

            #print(item)

            image_id = int(item['id'])
            category_id = 1 # TODO:  update how it is established
            tile_grid = self.grid[image_id-1:image_id]
            tile_vector = self.clip_vector(tile_grid)
            

            # Can be variable??
            if image_index == 0:
                width = item['width']
                height = item['height']
                image_index = -1

                #print("EXTRACTION", width, height)

            

            for index, polygon in enumerate(tile_vector.geometry):

                # Check polygon valid for class
                row = tile_vector.iloc[index]

                valid = True
                for col in self.class_column:
                    if not col in tile_vector.columns:
                        continue
                    annotated_class = str(row[col])
                    if (annotated_class in self.invalid_class):
                        valid = False
                        break

                if not valid:
                    continue
                elif isinstance(polygon, shapely.geometry.polygon.Polygon):

                    # Polygon
                    #print(polygon)
                    #print(type(polygon))
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
                        "iscrowd":0
                    }
                    ann_count = ann_count+1

                    annotations.append(ann)
                else:
                    print("INSTANCE", type(polygon))

            if self.progress_callback is not None:
                count = index
                total = len(self.coco_images)
                progress = count/total*0.4
                status = "processing"
                logs = "Initializing"
                info = {
                    "count": count
                    , "total": total
                    , "progress": progress
                    , "status": status
                    , "logs": logs
                }
                self.progress_callback(info)

        
        coco_dataset.dataset["annotations"] = annotations

        # Define categories (if not already defined in your annotations)
        categories = [
            {"id": 1, "name": self.category, "supercategory": self.supercategory},
            # ... add more categories
        ]
        coco_dataset.dataset["categories"] = categories

        # Save the COCO dataset as a JSON file
        #file_annotations = os.path.join(self.path_annotations,"annotations.json" )
        file_annotations = os.path.join(self.path_annotations,"instances_default.json" )
        #coco_dataset.save(file_annotations)

        with open(file_annotations, "w") as f:
            json.dump(coco_dataset.dataset, f, indent=4)

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


    # def create_grid(self, rows0):

    #     cols0 = rows0

    #     xmin, ymin, xmax, ymax = self.gdf.total_bounds

    #     tile_h = abs(ymax - ymin)/rows0
    #     tile_w = abs(xmax - xmin)/cols0

    #     cols = list(np.arange(xmin, xmax, tile_w))
    #     rows = list(np.arange(ymax, ymin, - tile_h))

    #     df_grid = pd.DataFrame()

    #     polygons = []
    #     count = 0
         
    #     for y in rows:
    #         for x in cols:
                       
    #             polygons.append(Polygon([(x,y), (x+tile_w, y), (x+tile_w, y- tile_h), (x, y- tile_h)]))

    #             df_item = pd.DataFrame({'id': count}, index=[count])
    #             df_grid = pd.concat((df_grid, df_item))

    #             count = count + 1

    #     #self.grid = gpd.GeoDataFrame(df_item, {'geometry':polygons})
    #     self.grid = gpd.GeoDataFrame(df_grid, geometry=polygons)

    #     # Fix index error with "module 'pandas' has no attribute 'Int64Index'"
    #     self.grid.reset_index(drop=True, inplace=True)
    #     self.grid.set_index("id", inplace = True)

    #     #self.grid["row_id"] = self.grid.index + 1
    #     #self.grid.reset_index(drop=True, inplace=True)
    #     #self.grid.set_index("row_id", inplace = True)

    #     self.grid = self.grid.set_crs(epsg=self.crs, allow_override=True)
    #     #grid.to_file("grid.shp")

    def create_grid(self, rows0, w_overlap=0, h_overlap=0):

        cols0 = rows0

        xmin, ymin, xmax, ymax = self.gdf.total_bounds

        Dh = abs(ymax - ymin)
        # oh = Dh*h_overlap
        # tile_h = (Dh - oh)/rows0 + oh

        tile_h = Dh/(rows0-rows0*h_overlap+h_overlap)
        oh = tile_h*h_overlap

        Dw = abs(xmax - xmin)
        # ow = Dw*w_overlap
        # tile_w = (Dw - ow)/cols0 + ow

        tile_w = Dw/(cols0-cols0*w_overlap+w_overlap)
        ow = tile_w*w_overlap


        # # Use only the value for the max 
        # if Dw > Dh and Dh/Dw > 0.7:
        #     tile_h = tile_w
        #     oh = ow
        # elif Dh > Dw and Dw/Dh > 0.7:
        #     tile_w = tile_h
        #     ow = oh

        # # Use only the value for the max 
        if Dw > Dh:
            tile_h = tile_w
            oh = ow
        elif Dh > Dw:
            tile_w = tile_h
            ow = oh

        #tile_h = abs(ymax - ymin)/rows0
        #tile_w = abs(xmax - xmin)/cols0

        #cols = list(np.arange(xmin, xmax, tile_w))
        #rows = list(np.arange(ymax, ymin, - tile_h))

        #print("TILE SIZE")
        #print(tile_h)
        #print(tile_w)

        cols = list(np.arange(xmin, xmax-(ow), tile_w - ow))
        rows = list(np.arange(ymax, ymin-(-(oh)), - (tile_h-oh)))

        #print(cols)
        #print(rows)

        df_grid = pd.DataFrame()

        polygons = []
        count = 0
         
        for y in rows:
            for x in cols:

                poly = Polygon([(x,y), (x+tile_w, y), (x+tile_w, y- tile_h), (x, y- tile_h)])


                #Check if polygon covers any annotation
                if not self.allow_no_annotations:

                    covered_items = []
                    for index, geom in enumerate(self.vector.geometry):
                        # print(poly)
                        # print(geom)
                        # print(index)
                        # print(poly.covers(geom))
                        if poly.covers(geom):
                            covered_items.append(geom)

                    #print(poly.covers(self.gdf.geometry))

                    if len(covered_items) == 0:
                        #print("No items are covered by the polygon.")
                        continue
                    #else:
                    #    print(f"{len(covered_items)} items")


                polygons.append(poly)

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

    # def create_grid(self, rows0, h_overlap=0, v_overlap=0):
    #     """
    #     Create a geospatial grid with overlapping tiles.
        
    #     Parameters:
    #     rows0: int
    #         Number of rows (and columns) for the grid.
    #     h_overlap: float, optional
    #         Horizontal overlap distance (in the same units as the CRS). 
    #         (For a raster, convert pixel overlap to map units using pixel resolution.)
    #     v_overlap: float, optional
    #         Vertical overlap distance (in the same units as the CRS).
    #     """
    #     # Define grid dimensions (square grid)
    #     cols0 = rows0

    #     # Get total bounds of the input GeoDataFrame
    #     xmin, ymin, xmax, ymax = self.gdf.total_bounds

    #     # Compute the nominal tile width and height (without overlap)
    #     tile_w = abs(xmax - xmin) / cols0
    #     tile_h = abs(ymax - ymin) / rows0

    #     # Calculate the step sizes (distance between the origins of adjacent tiles)
    #     # With a non-zero overlap, the step is smaller than the tile size.
    #     step_x = tile_w - tile_w*h_overlap
    #     step_y = tile_h - tile_h*v_overlap

    #     print(tile_w, h_overlap)
    #     print(tile_h, v_overlap)

    #     if step_x <= 0 or step_y <= 0:
    #         raise ValueError("Overlap is too large compared to the tile size.")

    #     # Generate the coordinates for the grid origins.
    #     # Note: np.arange might not include the very last cell if the bounds are not an exact multiple.
    #     cols = np.arange(xmin, xmax, step_x)
    #     rows = np.arange(ymax, ymin, -step_y)

    #     df_grid = pd.DataFrame()
    #     polygons = []
    #     count = 0

    #     for y in rows:
    #         for x in cols:
    #             # Create a polygon for each grid cell.
    #             # Each cell is defined to have the full tile_w x tile_h dimensions,
    #             # so adjacent cells will overlap by the specified amounts.
    #             poly = Polygon([
    #                 (x, y),
    #                 (x + tile_w, y),
    #                 (x + tile_w, y - tile_h),
    #                 (x, y - tile_h)
    #             ])
    #             polygons.append(poly)

    #             # Record the id for each cell.
    #             df_item = pd.DataFrame({'id': [count]})
    #             df_grid = pd.concat([df_grid, df_item], ignore_index=True)
    #             count += 1

    #     # Create a GeoDataFrame for the grid.
    #     self.grid = gpd.GeoDataFrame(df_grid, geometry=polygons)

    #     # Reset and set the index
    #     self.grid.reset_index(drop=True, inplace=True)
    #     self.grid.set_index("id", inplace=True)

    #     # Set the coordinate reference system (CRS)
    #     self.grid = self.grid.set_crs(epsg=self.crs, allow_override=True)

        

    def clip_raster(self, id, filename, scale = 1.0):

        #vector = self.grid.to_crs(self.raster.crs)
        #vector = self.grid[id:id+1].to_crs(self.raster.crs)
        safe_crs = CRS.from_user_input(self.raster.crs.toWkt())
        vector = self.grid[id:id+1]#.to_crs(safe_crs)
        print("SAFE_CRS:", safe_crs)
        wkt = self.raster.crs().toWkt()
        safe_crs2 = CRS.from_wkt(wkt)
        vector = vector.to_crs(safe_crs2)

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
        #tile, tile_transform = mask(raster, vector.geometry, crop=True, filled = True)
        tile, tile_transform = mask(raster, vector.geometry, crop=True)

        width = tile.shape[2]
        height = tile.shape[1]

        if "tif" in self.output_format:

            tile_meta = self.raster.meta.copy()

            tile_meta.update({
                "driver":"Gtiff",
                "height":height, # height starts with shape[1]
                "width":width, # width starts with shape[2]
                "transform":tile_transform
            })
            #print("TILE", tile.shape[1], tile.shape[2])
            with rio.open(filename, 'w', **tile_meta) as dst:
                dst.write(tile)

        else:
            
            tile = np.transpose(tile, (1,2,0))
            # print(type(tile))
            # print(tile.shape)
            # print(filename)
            tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)
            cv.imwrite(filename, tile)

        return tile, width, height

    def clip_vector(self, tile_grid):

        if self.allow_clipped_annotations:
            tile_vector = gpd.overlay(self.vector, tile_grid)
        else:
            # Identify geometries that do NOT intersect the tile grid boundaries
            vector_no_edge_intersect = self.vector[~self.vector.intersects(tile_grid.unary_union.boundary)]
            tile_vector = gpd.overlay(vector_no_edge_intersect, tile_grid)

        #print(type(self.grid.geometry[id]))

        #tile_vector = self.vector.clip(self.grid.geometry[id])
        #tile_vector = gpd.overlay(self.vector, tile_grid)

        # Save for debugging purposes
        # print(tile_vector)

        # tile_vector["row_id"] = tile_vector.index
        # tile_vector.reset_index(drop=True, inplace=True)
        # tile_vector.set_index("row_id", inplace = True)

        # tile_vector.to_file("D:/local_mydata/ROI/sample/tile_" + str(id) + ".shp" )

        return tile_vector
    
    def convert(self, path_output, rows = 1, scale = 1.0, overlap = 0):

        

        # if scale != 1.0:
        #     self.resample_raster()
        
        # # Configure the output folder structure
        self.set_path_output(path_output)
        self.create_output_folders()

        # Create a vector grid for each tile
        self.create_grid(rows, overlap, overlap)

        if self.progress_callback is not None:
            count = 0
            total = 0
            progress = 0
            status = "processing"
            logs = "Initializing"
            info = {
                "count": count
                , "total": total
                , "progress": progress
                , "status": status
                , "logs": logs
            }
            self.progress_callback(info)


        # Extract tiles and save
        self.extract_tiles()

        # # Extract annotations
        self.extract_annotations()

        # if not "tif" in self.output_format:
        #     # Find all .tif and .tiff files
        #     tif_files = glob.glob(os.path.join(self.path_images, "*.tif"))
        #     tiff_files = glob.glob(os.path.join(self.path_images, "*.tiff"))

        #     # Combine and delete
        #     for file_path in tif_files + tiff_files:
        #         os.remove(file_path)
        #         print(f"Deleted: {file_path}")

        if self.temp_file is not None:
            self.temp_file.close()

        if self.progress_callback is not None:
                count = 0
                total = 0
                progress = 100
                status = "finished"
                logs = "Finishing"
                info = {
                    "count": count
                    , "total": total
                    , "progress": progress
                    , "status": status
                    , "logs": logs
                }
                self.progress_callback(info)

        return