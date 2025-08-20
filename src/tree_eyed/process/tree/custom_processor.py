import os
import numpy as np
import json


def pos2coords(pos, extent, img_width, img_height):

        # print(extent)
        # print(extent.xMinimum())
        # print(extent.xMaximum())
        # print(extent.yMinimum())
        # print(extent.yMaximum())
        # print(extent.width())
        # print(extent.height())
        # print(self.iface.mapCanvas().mapSettings().destinationCrs().authid())

        # Solve extent is Bound from rasterio
        

        # coord_x_min = extent.xMinimum()
        # coord_y_min = extent.yMinimum()

        # coord_width = extent.width()
        # coord_height = extent.height()

        # Get xmin and ymin from extent
        coord_x_min = extent[0]
        coord_y_min = extent[1]

        # Get width and height from extent
        coord_width = extent[2] - extent[0]
        coord_height = extent[3] - extent[1]

 
        x = (pos[0])/img_width
        y = 1.0 - (pos[1])/img_height

        coord_x = x*coord_width + coord_x_min
        coord_y = y*coord_height + coord_y_min

        #res = (round(x*pixel_w, 2),round(y*pixel_h, 2))

        return (coord_x, coord_y)

def zonal_stats(gdf, raster_path, stats=['mean', 'min', 'max']):

    from osgeo import gdal
    from rasterio import features
    from rasterio.transform import Affine
    # Open raster
    ds = gdal.Open(raster_path)
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    arr = band.ReadAsArray()
    # Convert GDAL-style transform to Affine
    transform = Affine.from_gdal(*gt)
    results = []

    for idx, row in gdf.iterrows():
        # Rasterize the polygon to create a mask
        mask = np.zeros(arr.shape, dtype=np.uint8)
        shapes = [(row['geometry'], 1)]
        mask = features.rasterize(
            shapes,
            out_shape=arr.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Extract raster values within the mask
        masked = arr[mask == 1]
        stat = {}
        if 'mean' in stats:
            stat['mean'] = float(np.mean(masked))
        if 'min' in stats:
            stat['min'] = float(np.min(masked))
        if 'max' in stats:
            stat['max'] = float(np.max(masked))
        results.append(stat)
    return results

def save_shapefile_polygon_binary_raster(parameters):
        
        import cv2 as cv
        import pandas as pd
        import shapely
        import geopandas as gpd
        import rasterio as rio

        results = {}
        results["output_files"] = []

        binary_raster_path = parameters["binary_raster_path"]
        # Obtain extent, img_width, img_height, epsg from loading the binary raster with rio
        with rio.open(binary_raster_path) as src:
            extent = src.bounds
            img_width = src.width
            img_height = src.height
            epsg = src.crs.to_string()
            thresh = src.read(1)  # Read the first band

        #Check if thresh is CV_32FC1 and convert to CV_8UC1
        if thresh.dtype == np.float32:
            thresh = (thresh * 255).astype(np.uint8)
        
        #if 3 channels, convert to single channel
        if len(thresh.shape)==3 and thresh.shape[2]==3:
            thresh = cv.cvtColor(thresh, cv.COLOR_RGB2GRAY)
            

        df_tree_polygons_test = pd.DataFrame()
        tree_bb = []

        #(contours, hierarchy) = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Improve shape extraction
        (contours, hierarchy) = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)



        # Test improved shape extraction
        # processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_raster.tif")
        # crowns_mask, crowns_polygons= extract_individual_trees_opencv(processed_filepath, gaussian_sigma=2
        #                                                       , range_divisor=50
        #                                                       , min_crown_area=0
        #                                                       , max_crown_area=5000
        #                                                       )

        count = 0
        #contours
        for index,contour in enumerate(list(contours)):
            #print("contour ", index)
            new_contour = np.squeeze(contour)
            #print(new_contour)

            #print("number of dimensions")
            #print(new_contour.ndim)

            if new_contour.ndim < 2: #has only one point
                continue

            coord_polygon =[]
            for point in new_contour:

                coord = (point[0], point[1])
                new_coord = pos2coords(coord, extent, img_width, img_height)
                coord_polygon.append(new_coord)

            
            if len(coord_polygon) > 2:#at least 3 points

                polygon_object = shapely.geometry.Polygon(coord_polygon)
                polygon_object = shapely.make_valid(polygon_object)

                polygons = []
                if isinstance(polygon_object, shapely.geometry.polygon.Polygon):
                    polygons.append(polygon_object)
                elif isinstance(polygon_object, shapely.geometry.collection.GeometryCollection):
                    #polygons.append(polygon_object.geoms)
                    print("HERE")
                    print(polygon_object)
                    for item in polygon_object.geoms:
                        if isinstance(item, shapely.geometry.polygon.Polygon):
                            polygons.append(item)
                        elif isinstance(item, shapely.geometry.multipolygon.MultiPolygon): #include multipolygons
                            polygons.append(item)
                elif isinstance(polygon_object, shapely.geometry.multipolygon.MultiPolygon):#include multipolygons
                    polygons.append(polygon_object)
                else:
                    print(type(polygon_object))

                for polygon_geometry in polygons:

                    #print("polygon_geometry", polygon_geometry)


                    df_item_test = pd.DataFrame({'Class': 'Tree'
                                        , 'ID': count
                                        , 'label': 'Tree'
                                        }
                                        , index=[count])
                    df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))

                    #tree_bb.append(polygon_object)
                    tree_bb.append(polygon_geometry)

                    count = count + 1

        

        # Create geodataframe
        gdf_trees = gpd.GeoDataFrame(df_tree_polygons_test, geometry=tree_bb)
        gdf_trees = gdf_trees.set_crs(epsg=epsg.replace("EPSG:",""))
        
        #config_debug("len", len(gdf_trees))

        # Dissolve so no items in others
        gdf_trees = gdf_trees.dissolve()
        gdf_trees = gdf_trees.explode()
        # Solve problem additional columns
        #gdf_trees = gdf_trees.drop(columns=['level_0','level_1'])
        gdf_trees['ID'] = range(len(gdf_trees))

        #gdf_trees.to_file("D:/local_mydata/tree/results/vector/result_polygons.shp")

        #print(gdf_trees)
        # Fix geometries
        #gdf_trees = gdf_trees.make_valid()
        #print(gdf_trees)
        #gdf_trees['geometry'] = gdf_trees.geometry.apply(lambda x: x.make_valid())
        #gdf_trees['geometry'] = gdf_trees.make_valid()

        #test_df = gdf_trees.copy()
        #print(test_df.make_valid().info())

        # # Add additional

        # lon = extent.xMinimum()
        # lat = extent.yMinimum()

        # # Guarantee correct lat lon crs
        # #source_crs = layer.crs()
        # source_crs = QgsCoordinateReferenceSystem(epsg)
        # target_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        # transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

        # geom = QgsGeometry(QgsPoint(extent.xMinimum(), extent.yMinimum()))
        # geom.transform(transform)

        # lon = geom.constGet().x()
        # lat = geom.constGet().y()

        # Assume extent is a tuple: (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = extent

        # Use pyproj for coordinate transformation
        from pyproj import Transformer

        # Convert EPSG string to int if needed
        if isinstance(epsg, str) and epsg.startswith("EPSG:"):
            epsg_code = int(epsg.split(":")[1])
        else:
            epsg_code = int(epsg)

        # Transform lower-left corner to WGS84
        transformer = Transformer.from_crs(epsg_code, 4326, always_xy=True)
        lon, lat = transformer.transform(minx, miny)


        new_crs = "+proj=cea +lat_0=" + str(lat)  + " +lon_0="+ str(lon) + " +units=m"
        #print(new_crs)
        #df = df.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")
        gdf_trees["area_m2"] = gdf_trees.to_crs(new_crs).area
        gdf_trees["perim_m"] = gdf_trees.to_crs(new_crs).length
        gdf_trees["a_diam_m"] = np.sqrt(gdf_trees["area_m2"]*4.0/np.pi)

        # Add lat lon of centroid
        gdf_trees["lon"] = gdf_trees.geometry.centroid.x
        gdf_trees["lat"] = gdf_trees.geometry.centroid.y

        # Calculate circularity
        gdf_trees["circularity"] = 4 * np.pi * gdf_trees["area_m2"] / gdf_trees["perim_m"]**2

        # Add estimated height
        processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_raster.tif")
        if (os.path.exists(processed_filepath)): 
            stats = zonal_stats(gdf_trees, processed_filepath, stats=['mean', 'min', 'max'])
            gdf_trees['h_mean'] = [s.get('mean', None) for s in stats]
            gdf_trees['h_min'] = [s.get('min', None) for s in stats]
            gdf_trees['h_max'] = [s.get('max', None) for s in stats]

        
        #save
        if "polygons" in parameters["vector_outputs"]:
            output_filename = os.path.join(parameters["output_path"], parameters["prefix"] + "_vector.shp")
            gdf_trees.to_file(output_filename, index=False)
            
            results["output_files"].append(output_filename)

        if "bounding_boxes" in parameters["vector_outputs"]:
            output_filename = os.path.join(parameters["output_path"], parameters["prefix"] + "_vector_bb.shp")

            gdf_trees_bb = gdf_trees.copy()
            #gdf_trees_bb['geometry'] = gdf_trees_bb['geometry'].bounds
            bb = []
            for geom in gdf_trees_bb['geometry']:                
                bb.append(shapely.geometry.box(*geom.bounds))
                #print(geom.bounds)
            #print(len(bb))
            gdf_trees_bb['geometry'] = bb
            #gdf_trees_bb = gpd.GeoDataFrame(df_tree_polygons_test, geometry=bb)
            #print(gdf_trees_bb.info())

            gdf_trees_bb.to_file(output_filename, index=False)

            results["output_files"].append(output_filename)

        if "centroids" in parameters["vector_outputs"]:

            output_filename = os.path.join(parameters["output_path"], parameters["prefix"] + "_vector_centroids.shp")

            gdf_trees_c = gdf_trees.copy()
            gdf_trees_c['geometry'] = gdf_trees_c['geometry'].centroid

            gdf_trees_c.to_file(output_filename, index=False)

            results["output_files"].append(output_filename)

        return results

def bb_2_geodataframe(df, parameters):

        import cv2 as cv
        import pandas as pd
        import shapely
        import geopandas as gpd
        import rasterio as rio

        results = {}
        results["output_files"] = []

        has_epsg = False

        input_raster_path = parameters["input_raster_path"]
        # Obtain extent, img_width, img_height, epsg from loading the binary raster with rio
        # Check if input is a GeoTIFF, otherwise use default values
        try:
            with rio.open(input_raster_path) as src:
                extent = src.bounds
                img_width = src.width
                img_height = src.height
                epsg = src.crs.to_string()
                has_epsg = True
        except Exception:
            # Not a GeoTIFF or cannot read spatial info
            import PIL.Image
            img = PIL.Image.open(input_raster_path)
            img_width, img_height = img.size
            # Default extent: (0, 0, width, height)
            extent = (0, 0, img_width, img_height)
            epsg = None
            has_epsg = False

        print(type(df))
        if type(df) == "NoneType":
            print("No results")
            return

        df_tree_polygons_test = pd.DataFrame()
        tree_bb = []
        count = 0

        for index, detection in df.iterrows():

            xmin = detection["xmin"]
            ymin = detection["ymin"]
            xmax = detection["xmax"]
            ymax = detection["ymax"]

            new_contour = []
            new_contour.append((xmin,ymin))
            new_contour.append((xmax,ymin))
            new_contour.append((xmax,ymax))
            new_contour.append((xmin,ymax))

            coord_polygon =[]
            for point in new_contour:

                coord = (point[0], point[1])
                new_coord = pos2coords(coord, extent, img_width, img_height)

                if not has_epsg:
                    new_coord = (new_coord[0],new_coord[1]-img_height)

                coord_polygon.append(new_coord)

            #print("len", len(coord_polygon))

            if len(coord_polygon) > 2:#at least 3 points

                polygon_object = shapely.geometry.Polygon(coord_polygon)

                df_item_test = pd.DataFrame({'Class': 'Tree'
                                    , 'ID': count
                                    , 'label': 'Tree'
                                    }
                                    , index=[count])
                df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))

                #tree_bb.append(polygon_object)
                tree_bb.append(polygon_object)

                count = count + 1

        # Create geodataframe
        gdf_trees = gpd.GeoDataFrame(df_tree_polygons_test, geometry=tree_bb)
        


        # Dissolve so no items in others
        #gdf_trees = gdf_trees.dissolve()
        #gdf_trees = gdf_trees.explode()

        
        # Add additional

        # lon = extent.xMinimum()
        # lat = extent.yMinimum()

        # # Guarantee correct lat lon crs
        # #source_crs = layer.crs()
        # source_crs = QgsCoordinateReferenceSystem(epsg)
        # target_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        # transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

        # geom = QgsGeometry(QgsPoint(extent.xMinimum(), extent.yMinimum()))
        # geom.transform(transform)

        # lon = geom.constGet().x()
        # lat = geom.constGet().y()

        if has_epsg:

            gdf_trees = gdf_trees.set_crs(epsg=epsg.replace("EPSG:",""))

            print(gdf_trees)

            

            # Assume extent is a tuple: (minx, miny, maxx, maxy)
            minx, miny, maxx, maxy = extent

            # Use pyproj for coordinate transformation
            from pyproj import Transformer

            # Convert EPSG string to int if needed
            if isinstance(epsg, str) and epsg.startswith("EPSG:"):
                epsg_code = int(epsg.split(":")[1])
            else:
                epsg_code = int(epsg)

            # Transform lower-left corner to WGS84
            transformer = Transformer.from_crs(epsg_code, 4326, always_xy=True)
            lon, lat = transformer.transform(minx, miny)

            new_crs = "+proj=cea +lat_0=" + str(lat)  + " +lon_0="+ str(lon) + " +units=m"
            #print(new_crs)
            #df = df.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")
            gdf_trees["area_m2"] = gdf_trees.to_crs(new_crs).area
            gdf_trees["a_diam_m"] = np.sqrt(gdf_trees["area_m2"]*4.0/np.pi)

        return gdf_trees



def export_coco_dataset(parameters, progress_callback = None, interruption_check = None):

    
    image_path = parameters["image_path"]
    annotations_path = parameters["annotations_path"]
    num_tiles = parameters["num_tiles"] # now it would be max pixels per tile
    overlap = int(parameters["overlap"])/100.0
    output_format = "." + parameters["output_format"].lower()
    
    dir_name = parameters["prefix"] + "_coco_dataset"
    
    path_output = os.path.join(parameters["output_path"], dir_name)

    #from .process.tree.qgis2coco.qgis2coco import QGIS2COCO
    from .qgis2coco.qgis2coco import QGIS2COCO
    from .qgis2coco.qgis2coco import check_raster


    metadata_final = check_raster(image_path)
    w = metadata_final["width"]
    h = metadata_final["height"]

    max_px = num_tiles

    rows = 1

    if (w > max_px or h > max_px):

        max_val = max(metadata_final["width"], metadata_final["height"])
        rows = np.ceil((max_val-overlap*max_px)/(max_px*(1-overlap)))

    COCO_CONTRIBUTOR = "TreeEyed Plugin | Tropical Forages Program | Alliance Bioversity International & CIAT"
    COCO_LICENSE = "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"
    COCO_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
    COCO_INFORMATION = ""

    exporter = QGIS2COCO(image_path
            , annotations_path
            , allow_clipped_annotations = False
            , allow_no_annotations = False
            , information = COCO_INFORMATION
            , license = COCO_LICENSE
            , license_url = COCO_LICENSE_URL
            , contributor = COCO_CONTRIBUTOR
            , output_format = output_format
            , progress_callback = progress_callback
            , interruption_check = interruption_check
        )
    exporter.convert(path_output, rows = rows, overlap = overlap)

def clean_cache_folder(output_dir):

    from .interface.cachemanager import CacheManager
    cache_manager = CacheManager(project_path = output_dir)
    cache_manager.clean_cache_folder()

def is_geotif(filepath):
    import rasterio as rio
    try:
        with rio.open(filepath) as src:
            # Check if the file is a TIFF and has a CRS (georeferencing)
            return src.driver == 'GTiff' and src.crs is not None
    except Exception:
        return False
    
def inference(parameters, progress_callback = None, interruption_check = None):
    print("Running inference task...")

    input_raster_path = parameters["input_raster_path"]

    is_geotif_flag = is_geotif(input_raster_path)

    if is_geotif_flag:
        return inference_georaster(parameters, progress_callback, interruption_check)
    else:
        return inference_img(parameters, progress_callback, interruption_check)

def inference_img(parameters, progress_callback = None, interruption_check = None):

    print("Running inference task for img...")

    input_raster_path = parameters["input_raster_path"]
    tile_size = parameters.get("tile_size", 1024)

    model = parameters["model"]

    if not 'tile_size' in parameters:
    
        if model == 'HighResCanopyHeight':
            tile_size = 256
        elif model == 'DeepForest':
            tile_size = 400
    # elif model == 'Mask R-CNN':
        elif model == "VHRTrees":
            tile_size = 960
        # elif model == "Custom ONNX Model":

    else:
        tile_size = parameters.get("tile_size", 1024)


    output_path = parameters["output_path"]
    prefix = parameters["prefix"]

    # Use cache folder
    # if the folder has the same input_raster_path, is not temporal (current view), and has the same tile_size, then use the same cache folder
    cache_key = {
        "input_raster_path": input_raster_path,
        "tile_size": tile_size,
        "is_temporal": parameters.get("is_temporal", False)
    }
    if (parameters.get("is_temporal", False)):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_key["timestamp"] = timestamp

    from .interface.cachemanager import CacheManager
    cache_manager = CacheManager(project_path = output_path)
    key = cache_manager.compute_key(cache_key)
    
    outputdir = cache_manager.get_cache_folder_path("inference", key)
    print(f"Using cache folder: {outputdir}")

    tiles_dir = os.path.join(outputdir, "tiles")
    model = parameters["model"]
    model_processsed_folder = model.replace(" ", "_").lower()
    if "Custom ONNX Model" in model:
        model_processsed_folder = model_processsed_folder + os.path.basename(parameters["custom_model_filepath"]).replace(".onnx", "").replace(" ", "_")
    processed_dir = os.path.join(outputdir, "processed", model_processsed_folder)
    metadata_filepath = os.path.join(outputdir, "metadata.json")

    # Create output directories if they do not exist
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)    

    metadata_filepath = os.path.join(outputdir, "metadata.json") 

    if not os.path.exists(metadata_filepath):
        # Create json file to save metadata
        #metadata json includes input_raster_path only
        metadata = {
            "input_raster_path": input_raster_path,
            "prefix": prefix
            , "tile_size": tile_size
            , "is_temporal": parameters.get("is_temporal", False)
        }
        # Save metadata to a json file
        metadata_filepath = os.path.join(outputdir, "metadata.json")
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f)

    # Tile image using tile_size and opencv
    import cv2 as cv

    img = cv.imread(input_raster_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_raster_path}")
    h, w = img.shape[:2]
    stride = tile_size  # No overlap for now

    prefix = ''
    output_format = '.tif'


    # Calculate total number of tiles
    n_tiles_x = int(np.ceil(w / tile_size))
    n_tiles_y = int(np.ceil(h / tile_size))
    total_tiles = n_tiles_x * n_tiles_y

    tile_count = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]

            # Pad tile if it's smaller than tile_size
            tile_h, tile_w = tile.shape[:2]
            if tile_h < tile_size or tile_w < tile_size:
                if len(tile.shape) == 3:
                    padded_tile = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
                else:
                    padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded_tile[:tile_h, :tile_w] = tile
                tile = padded_tile


            basename = f"{prefix}{tile_count:05d}{output_format}"
            tile_filename = os.path.join(tiles_dir, basename)
            
            cv.imwrite(tile_filename, tile)
            tile_count += 1

            if progress_callback is not None:
                count = tile_count
                total = total_tiles
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
                progress_callback(info)

            if interruption_check is not None:
                if interruption_check():
                    break

    print(f"Tiled image into {tile_count} tiles at {tiles_dir}")


    # Process
    pattern = os.path.join(tiles_dir, "*.tif")
    import glob
    tiles = glob.glob(pattern)

    parameters.update(
        {
            "processed_dir" : processed_dir
            , "tiles_dir": tiles_dir
            , "tiles": tiles
        }
    )

    # Batch process, TODO: parallelize
    results = model_inference(parameters, progress_callback, interruption_check)
    

    # Merge

    if "tiles_processed" in results and len(results["tiles_processed"]) > 0:

        # Check if merge tif raster or vector shp
        first = results["tiles_processed"][0]
        if first.endswith(".shp"):

            processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_bb.shp")

            print(processed_filepath)

            # Merge shp filepaths in results["tiles_processed"]
            import geopandas as gpd
            import pandas as pd
            import re

            gdfs = []
            for shp_path in results["tiles_processed"]:
                gdf = gpd.read_file(shp_path)
                gdfs.append(gdf)


            n_tiles_x = int(np.ceil(w / tile_size))
            n_tiles_y = int(np.ceil(h / tile_size))

            # Helper to get tile index from filename
            def get_tile_index(shp_path):
                # Assumes filename like '00001.shp' or 'prefix00001.shp'
                base = os.path.basename(shp_path)
                match = re.search(r"(\d+)\.shp$", base)
                return int(match.group(1)) if match else -1

            # Sort shapefiles by tile index
            tile_shps = sorted(results["tiles_processed"], key=get_tile_index)



            # Concatenate all GeoDataFrames
            #merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
            merged_gdfs = []
            for idx, shp_path in enumerate(tile_shps):
                gdf = gpd.read_file(shp_path)
                if gdf.empty:
                    continue
                tile_idx = get_tile_index(shp_path)
                row = tile_idx // n_tiles_x
                col = tile_idx % n_tiles_x
                x_offset = col * tile_size
                y_offset = row * tile_size

                # Shift geometries by tile offset
                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].translate(xoff=x_offset, yoff=-y_offset)
                merged_gdfs.append(gdf)
            
            merged_gdf = gpd.GeoDataFrame(pd.concat(merged_gdfs, ignore_index=True))


            # Dissolve to merge geometries
            #merged_gdf = merged_gdf.dissolve()
            #merged_gdf = merged_gdf.explode()
            # Save merged GeoDataFrame to a new shapefile
            merged_gdf = merged_gdf.reset_index(drop=True)
            merged_gdf.to_file(processed_filepath, driver='ESRI Shapefile')

            parameters["output_files"] = []
            parameters["output_files"].append(processed_filepath)

            

        elif first.endswith(".tif"):

            import re
            # Merge tiles using OpenCV based on enumeration in their filenames

            # Get all tile files and sort by enumeration in filename
            tile_files = glob.glob(os.path.join(processed_dir, f"*{output_format}"))
            def tile_sort_key(path):
                # Extract the number from the filename (e.g., 00001 from prefix00001.tif)
                match = re.search(r"(\d+)\.\w+$", os.path.basename(path))
                return int(match.group(1)) if match else -1
            tile_files = sorted(tile_files, key=tile_sort_key)

            # Calculate number of tiles in x and y direction
            n_tiles_x = int(np.ceil(w / tile_size))
            n_tiles_y = int(np.ceil(h / tile_size))

            # Read all tiles into a list
            tiles = [cv.imread(f, cv.IMREAD_UNCHANGED) for f in tile_files]

            # Determine tile shape (handle last tiles which may be smaller)
            tile_shapes = [t.shape for t in tiles]
            tile_h, tile_w = tile_shapes[0][:2]
            channels = tile_shapes[0][2] if len(tile_shapes[0]) == 3 else 1

            # Prepare empty canvas for merged image
            if channels == 1:
                merged = np.zeros((h, w), dtype=tiles[0].dtype)
            else:
                merged = np.zeros((h, w, channels), dtype=tiles[0].dtype)

            # Place each tile in the correct position
            for idx, tile in enumerate(tiles):
                row = idx // n_tiles_x
                col = idx % n_tiles_x
                y0 = row * tile_size
                x0 = col * tile_size
                y1 = min(y0 + tile.shape[0], h)
                x1 = min(x0 + tile.shape[1], w)
                merged[y0:y1, x0:x1] = tile[:y1-y0, :x1-x0]

            # Save merged image
            parameters["output_files"] = []

            # Model output
            processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_raster.tif")
            parameters["output_files"].append(processed_filepath)

            cv.imwrite(processed_filepath, merged)

    return parameters

def polygon_nms(gdf, iou_threshold=0.5, score_col=None):
    # If thre is a confidence score, sort by it (descending)
    if score_col and score_col in gdf.columns:
        gdf = gdf.sort_values(score_col, ascending=False).reset_index(drop=True)
    else:
        gdf = gdf.reset_index(drop=True)
    
    keep = []
    suppressed = set()
    for i, poly_i in enumerate(gdf.geometry):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i+1, len(gdf)):
            if j in suppressed:
                continue
            poly_j = gdf.geometry[j]
            inter = poly_i.intersection(poly_j).area
            union = poly_i.union(poly_j).area
            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                suppressed.add(j)
    return gdf.iloc[keep].reset_index(drop=True)

def inference_georaster(parameters, progress_callback = None, interruption_check = None):

    print("Running inference task for georaster...")

    input_raster_path = parameters["input_raster_path"]
    tile_size = parameters.get("tile_size", 1024)

    model = parameters["model"]
    if model == 'HighResCanopyHeight':
        tile_size = 256
    elif model == 'DeepForest':
        tile_size = 400
   # elif model == 'Mask R-CNN':
    elif model == "VHRTrees":
        tile_size = 960
    # elif model == "Custom ONNX Model":

    output_path = parameters["output_path"]
    prefix = parameters["prefix"]

    # Use cache folder
    # if the folder has the same input_raster_path, is not temporal (current view), and has the same tile_size, then use the same cache folder
    cache_key = {
        "input_raster_path": input_raster_path,
        "tile_size": tile_size,
        "is_temporal": parameters.get("is_temporal", False)
    }
    if (parameters.get("is_temporal", False)):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_key["timestamp"] = timestamp

    from .interface.cachemanager import CacheManager
    cache_manager = CacheManager(project_path = output_path)
    key = cache_manager.compute_key(cache_key)
    
    outputdir = cache_manager.get_cache_folder_path("inference", key)
    print(f"Using cache folder: {outputdir}")


    # CACHE_FOLDER = "_cache_tree_eyed"

    # cache_dir = os.path.join(output_path, CACHE_FOLDER)
    # os.makedirs(cache_dir, exist_ok=True)

    # import datetime
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # #dir_name = prefix + "_temp_dir_" + timestamp
    # dir_name = prefix + "_" + timestamp
    # outputdir = os.path.join(output_path, CACHE_FOLDER, dir_name)

    # # Check all metadata.json if it exists in folders in cache_dir
    # # if input_raster_path in metadata.json then break and assign parent folder to dir_name
    # for root, dirs, files in os.walk(cache_dir):
    #     if "metadata.json" in files:
    #         metadata_filepath = os.path.join(root, "metadata.json")
    #         # if file exists, read it and check if input_raster_path matches
    #         if os.path.exists(metadata_filepath):
    #             with open(metadata_filepath, "r") as f:
    #                 metadata = json.load(f)
    #                 # Compare file paths using os.path.abspath and os.path.normcase for cross-platform compatibility
    #                 meta_path = os.path.normcase(metadata.get("input_raster_path"))
    #                 raster_path = os.path.normcase(os.path.abspath(input_raster_path))

    #                 print(meta_path)
    #                 print(raster_path)

    #                 if meta_path == raster_path and not metadata.get("is_temporal",False):

    #                     if (metadata.get("tile_size") == tile_size):
    #                         dir_name = os.path.basename(root)
    #                         outputdir = root
    #                         print(f"Found cached files for {input_raster_path} in {metadata_filepath}")
    #                         break


    tiles_dir = os.path.join(outputdir, "tiles")
    model = parameters["model"]
    model_processsed_folder = model.replace(" ", "_").lower()
    if "Custom ONNX Model" in model:
        model_processsed_folder = model_processsed_folder + os.path.basename(parameters["custom_model_filepath"]).replace(".onnx", "").replace(" ", "_")
    processed_dir = os.path.join(outputdir, "processed", model_processsed_folder)
    metadata_filepath = os.path.join(outputdir, "metadata.json")

    # Create output directories if they do not exist
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)    

    metadata_filepath = os.path.join(outputdir, "metadata.json") 

    if not os.path.exists(metadata_filepath):
        # Create json file to save metadata
        #metadata json includes input_raster_path only
        metadata = {
            "input_raster_path": input_raster_path,
            "prefix": prefix
            , "tile_size": tile_size
            , "is_temporal": parameters.get("is_temporal", False)
        }
        # Save metadata to a json file
        metadata_filepath = os.path.join(outputdir, "metadata.json")
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f)

    # Resample if necessary


    # TODO: Parallelize tiling and inference
    # Tile if necessary
    from .qgis2coco.qgis2coco import TILER    

    # tiling
    converter = TILER(input_raster_path
            , ""
            , category = "category"
            , supercategory = "supercategory"
            , allow_clipped_annotations = False
            , allow_no_annotations = False
            , class_column = ["label"]
            , invalid_class=["target", "empty"]
            , preffix = ''
            , crs = "4326"
            , progress_callback = progress_callback
            , interruption_check = interruption_check
            )
    
    converter.path_images = tiles_dir   
    
    from .qgis2coco.qgis2coco import check_raster
    metadata = check_raster(input_raster_path)


    w = metadata["width"]
    h = metadata["height"]
    max_px = tile_size
    #max_px = 256    
    overlap = 0

    # rows = 1

    # if (w > max_px or h > max_px):

    #     max_val = max(w,h)
    #     #print(max_val)        
    #     #rows = np.ceil(max_val/final_max_px)
    #     #rows = (max_val - np.ceil(overlap*max_val))/(max_px - np.ceil(overlap*max_val))
    #     rows = np.ceil((max_val-overlap*max_px)/(max_px*(1-overlap)))

    # print("rows", rows)
    # print("overlap", overlap)

    # # Create a vector grid for each tile
    # converter.create_grid(rows, overlap, overlap)

    converter.create_grid_px(max_px, overlap)  # Use 1 row for single tile extraction

    # Extract tiles and save
    converter.extract_tiles()


    pattern = os.path.join(tiles_dir, "*.tif")
    import glob
    tiles = glob.glob(pattern)


    parameters.update(
        {
            "processed_dir" : processed_dir
            , "tiles_dir": tiles_dir
            , "tiles": tiles
        }
    )



    # Batch process, TODO: parallelize
    results = model_inference(parameters, progress_callback, interruption_check)

    # Merge
    if "tiles_processed" in results and len(results["tiles_processed"]) > 0:

        # Check if merge tif raster or vector shp
        first = results["tiles_processed"][0]
        if first.endswith(".shp"):

            processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_bb.shp")

            print(processed_filepath)

            # Merge shp filepaths in results["tiles_processed"]
            import geopandas as gpd
            import pandas as pd

            gdfs = []
            for shp_path in results["tiles_processed"]:
                gdf = gpd.read_file(shp_path)
                gdfs.append(gdf)

            # Concatenate all GeoDataFrames
            merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

            # Dissolve to merge geometries
            #merged_gdf = merged_gdf.dissolve()
            #merged_gdf = merged_gdf.explode()
            # Save merged GeoDataFrame to a new shapefile
            merged_gdf = merged_gdf.reset_index(drop=True)

            # If model is VHRTrees apply nms
            if model == "VHRTrees":
                merged_gdf = polygon_nms(merged_gdf, iou_threshold=0.5, score_col=None)

            merged_gdf.to_file(processed_filepath, driver='ESRI Shapefile')


            # non-max suppression


            parameters["output_files"] = []
            parameters["output_files"].append(processed_filepath)



        
        elif first.endswith(".tif"):

            processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_raster.tif")

            print(processed_filepath)

            import rasterio as rio
            from rasterio.merge import merge

            srcs = [rio.open(p) for p in results["tiles_processed"]]

            mosaic, out_transform = merge(
                srcs,
                method="max",        # alternatives: "last", "min", "max", numpy.mean
                nodata=srcs[0].nodata, # keeps NoData consistent
                precision=10           # rounding in the affine transform (optional)
            )

            with rio.open(input_raster_path) as ref:
                from rasterio.windows import from_bounds
                # window covering original's bounds, expressed in mosaic pixel coords
                win = from_bounds(*ref.bounds, transform=out_transform)
                win = win.round_offsets().round_lengths()  # ensure integer indices

                r0, c0 = int(win.row_off), int(win.col_off)
                h,  w  = int(win.height),  int(win.width)

                cropped = mosaic[:, r0:r0+h, c0:c0+w]
                transform_cropped = rio.windows.transform(win, out_transform)

                # 3) Build output profile (keep your original creation options if any)
                meta = srcs[0].meta.copy()
                meta.update(
                    height=h,
                    width=w,
                    transform=transform_cropped,
                    count=cropped.shape[0]
                    # optionally keep compression/etc:
                    # compress='deflate', tiled=True, predictor=2
                )
                
                with rio.open(processed_filepath, "w", **meta) as dst:
                    dst.write(cropped)

            # meta = srcs[0].meta.copy()
            # meta.update(
            #     height=mosaic.shape[1],
            #     width=mosaic.shape[2],
            #     transform=out_transform,
            #     count=mosaic.shape[0]   # number of bands
            # )


            # with rio.open(processed_filepath, "w", **meta) as dst:
            #     dst.write(mosaic)



            # if progress_callback is not None:
            #         count = index+1
            #         total = total_tiles
            #         progress = count/total
            #         status = "processing"
            #         logs = "Inference progress..."
            #         info = {
            #             "count": count
            #             , "total": total
            #             , "progress": progress
            #             , "status": status
            #             , "logs": logs
            #         }
            #         progress_callback(info)

            #     if interruption_check is not None:
            #         if interruption_check():
            #             break

            parameters["output_files"] = []

            # Model output
            processed_filepath = os.path.join(parameters["output_path"], parameters["prefix"] + "_raster.tif")
            parameters["output_files"].append(processed_filepath)


            # Add other outputs

            if "binary" in parameters["raster_outputs"]:

                binary_path = processed_filepath.replace("_raster", "_raster_binary")
                parameters["binary_raster_path"] = binary_path

                import cv2 as cv

                # Load the processed raster
                cv_img = cv.imread(processed_filepath, cv.IMREAD_UNCHANGED)

                print(cv_img.dtype)
                print(cv_img.shape)

                max_value = np.max(cv_img)
                min_value = np.min(cv_img)

                range = max_value - min_value
                interval = range / 500
                #threshold = min_value + interval * parameters["hrch_threshold"]
                threshold = min_value + interval

                #value = parameters["hrch_threshold"]*max_value
                #pred_binary = (cv_img > value)*1.0
                pred_binary = (cv_img >= threshold)*1.0

                print(cv_img.dtype)
                print(cv_img.shape)
                print(np.unique(pred_binary))
                print("max value", max_value)
                #print("value", value)

                # crowns_mask, crowns_polygons= extract_individual_trees_opencv(processed_filepath, gaussian_sigma=2
                #                                               , range_divisor=50
                #                                               , min_crown_area=0
                #                                               , max_crown_area=5000
                #                                               )
                
                


                np2tif_2(pred_binary, processed_filepath, binary_path, rio.float32)
                #watershed_path = binary_path.replace("_raster_binary","_raster_binary_watershed")
                #np2tif_2(crowns_mask, processed_filepath, watershed_path, rio.uint8)

                #add result filepath to list of results
                parameters["output_files"].append(binary_path)
                #parameters["output_files"].append(watershed_path)


            if len(parameters["vector_outputs"]) > 0:

                # Generate vector outputs
                results = save_shapefile_polygon_binary_raster(parameters)

                # cocatenate results output files with parameters["output_files"]
                parameters["output_files"].extend(results["output_files"])

            #if "centroids" in self.parameters["vector_outputs"]:

        else:
            print("Unsupported file type for merging. Expected .tif or .shp files.")


    # Post-process results

    
    


    # # final handling if its temporal
    # if parameters.get("is_temporal", False):
    #     cache_manager.clean_cache_folder_path("inference", key)
    #     cache_manager.remove_temp_raster()



    return parameters

def postprocess_crowns(contours, min_area=15, max_area=500, min_circularity=0.5, overlap_thresh=0.5):
    """
    Postprocess tree crown contours:
    - Filter by area
    - Filter by circularity
    - Merge overlapping crowns

    Returns:
        filtered_contours: list of contours after filtering and merging
    """
    import cv2 as cv
    import numpy as np

    # Step 1: Area and circularity filtering
    filtered = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Convexity check
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0 or area / hull_area < 0.7:
            continue  # discard non-convex shapes

        filtered.append(cnt)

    # Step 2: Merge overlapping crowns (simple bounding box IoU)
    def bbox_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    merged = []
    used = [False] * len(filtered)
    for i, cntA in enumerate(filtered):
        if used[i]:
            continue
        boxA = cv.boundingRect(cntA)
        xA, yA, wA, hA = boxA
        boxA = [xA, yA, xA + wA, yA + hA]
        merged_cnt = cntA.copy()
        for j, cntB in enumerate(filtered):
            if i == j or used[j]:
                continue
            boxB = cv.boundingRect(cntB)
            xB, yB, wB, hB = boxB
            boxB = [xB, yB, xB + wB, yB + hB]
            if bbox_iou(boxA, boxB) > overlap_thresh:
                merged_cnt = np.vstack([merged_cnt, cntB])
                used[j] = True
        merged.append(merged_cnt)
        used[i] = True

    return merged

def extract_individual_trees_opencv(
    chm_path,
    min_height=None,
    gaussian_sigma=2,
    min_crown_area=None,
    max_crown_area=None,
    range_divisor=50,
    min_circularity=0.5,
    overlap_thresh=0.5
):
    """
    Extract individual tree crowns from a CHM raster using OpenCV and NumPy.

    Args:
        chm_path (str): Path to the CHM raster (GeoTIFF or image).
        min_height (float or None): Minimum canopy height to consider as tree (meters). If None, use dynamic threshold.
        gaussian_sigma (float): Sigma for Gaussian smoothing.
        min_crown_area (int): Minimum area (pixels) for a crown to be kept.
        max_crown_area (int or None): Maximum area (pixels) for a crown to be kept. If None, no upper limit.
        range_divisor (int): Divides the height range to set dynamic threshold if min_height is None.

    Returns:
        crowns_mask (np.ndarray): Labeled mask of individual crowns.
        crowns_polygons (list): List of polygons (contours) for each crown.
    """

    crowns_mask = [] 
    crowns_polygons = []

    import cv2 as cv

    # Load CHM
    chm = cv.imread(chm_path, cv.IMREAD_UNCHANGED)
    if chm is None:
        raise ValueError(f"Could not read CHM: {chm_path}")

    # Smooth the CHM
    chm_smooth = cv.GaussianBlur(chm, (0, 0), gaussian_sigma)
    #chm_smooth = chm

    # Dynamic thresholding based on height range
    max_value = np.max(chm_smooth)
    min_value = np.min(chm_smooth)
    if min_height is None:
        interval = (max_value - min_value) / range_divisor
        threshold = min_value + interval
    else:
        threshold = min_height    


    chm_smooth = chm_smooth*(chm_smooth >= (min_value + interval))


    # Threshold the CHM
    canopy_mask = (chm_smooth >= threshold).astype(np.uint8)
    print(np.unique(canopy_mask))

    # Detect local maxima (tree tops) using dilation
    neighborhood = np.ones((3, 3), dtype=np.uint8)
    local_max = cv.dilate(chm_smooth, neighborhood)
    peaks_mask = (chm_smooth == local_max) & (chm_smooth >= threshold) & (canopy_mask == 1)

    # # plt.tight_layout()
    # # plt.show()

    # Label local maxima as markers
    markers = np.zeros_like(chm_smooth, dtype=np.int32)
    peak_indices = np.argwhere(peaks_mask)
    for idx, (y, x) in enumerate(peak_indices):
        markers[y, x] = idx + 1  # Unique marker for each tree top

    canopy_mask_inverted = cv.bitwise_not(canopy_mask)


    # #print(markers.shape)
    # #print(markers)

    # # # Visualize markers
    # # plt.figure(figsize=(12, 6))
    # # plt.title("Watershed Markers")
    # # plt.imshow(markers, cmap="nipy_spectral")
    # # plt.axis("off")
    # # plt.show()

    # # Invert the CHM for watershed
    if np.min(chm_smooth) < 0:
        chm_smooth = chm_smooth - np.min(chm_smooth)
    chm_smooth = chm_smooth*canopy_mask


    #chm_inverted = (np.max(chm_smooth) - chm_smooth).astype(np.uint8)

    #**************

    # Normalize CHM and distance transform
    norm_chm = chm_smooth.astype(np.float32)
    norm_chm = (norm_chm - norm_chm.min()) / (norm_chm.max() - norm_chm.min())

    # Distance transform from canopy mask
    dist_transform = cv.distanceTransform((canopy_mask * 255).astype(np.uint8), cv.DIST_L2, 5)
    norm_dist = dist_transform / (dist_transform.max() + 1e-6)

    # Combine both (α = CHM, β = distance)
    alpha, beta = 0.6, 0.1
    combined = alpha * (1 - norm_chm) + beta * (1 - norm_dist)
    chm_inverted = (combined * 255).astype(np.uint8)


    # # Prepare mask for watershed (must be 8-bit single channel)
    #mask = (canopy_mask * 255).astype(np.uint8)

    # Watershed expects a 3-channel image
    #chm_color = cv.cvtColor(canopy_mask_inverted, cv.COLOR_GRAY2BGR)
    chm_color = cv.cvtColor(chm_inverted, cv.COLOR_GRAY2BGR)

    # # # Visualize markers
    # # plt.figure(figsize=(12, 6))
    # # plt.title("Watershed Markers")
    # # plt.imshow(markers, cmap="nipy_spectral")
    # # plt.axis("off")
    # # plt.show()
    # # print(np.min(markers))
    # # print(np.max(markers))

    # Apply marker-controlled watershed
    cv.watershed(chm_color, markers)


    # # Post-process: remove small/large segments and extract polygons
    # crowns_mask = np.zeros_like(markers, dtype=np.uint16)
    # crowns_polygons = []
    valid_countours = []
    for label in np.unique(markers):
        if label <= 0:
            continue
        mask_label = (markers == label).astype(np.uint8)
        area = np.sum(mask_label)
        #print(area)
        if min_crown_area is not None and area < min_crown_area:
            #print(f"area is smaller than {min_crown_area}")
            continue
        if max_crown_area is not None and area > max_crown_area:
            #print(f"area is larger than {max_crown_area}")
            continue
        #crowns_mask[markers == label] = label
        contours, _ = cv.findContours(mask_label, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 2:
                valid_countours.append(cnt)

    # Postprocess polygons
    crowns_polygons = postprocess_crowns(
        valid_countours,
        min_area=min_crown_area,
        max_area=max_crown_area,
        min_circularity=min_circularity,
        overlap_thresh=overlap_thresh
    )

    # Create a new mask for postprocessed crowns
    crowns_mask = np.zeros_like(markers, dtype=np.uint16)
    for idx, cnt in enumerate(crowns_polygons, start=1):
        cv.drawContours(crowns_mask, [cnt], -1, idx, thickness=-1)


    return crowns_mask, crowns_polygons


def tiling(parameters, progress_callback = None, interruption_check = None):
    """
    Tiling function to split large images into smaller tiles.
    """
    from .tree_predictor_task import TILER

    # Get basename without extension
    basename = os.path.splitext(os.path.basename(output_filepath))[0]

    # Set output folder as filepath dir
    output_folder = os.path.dirname(output_filepath)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_folder = os.path.join(output_folder, f"{basename}_{timestamp}")
    output_folder = os.path.join(output_folder, f"{basename}_foragesrois_temp")
    images_dir = os.path.join(output_folder, "tiles")
    shp_dir = os.path.join(output_folder, "shp")
    

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(shp_dir, exist_ok=True)

    # tiling
    converter = TILER(input_filepath
            , ""
            , category = "category"
            , supercategory = "supercategory"
            , allow_clipped_annotations = False
            , allow_no_annotations = False
            , class_column = ["label"]
            , invalid_class=["target", "empty"]
            , preffix = ''
            , crs = "4326"
            )
    
    converter.path_images = images_dir
    
    metadata = check_raster(input_filepath)


    w = metadata["width"]
    h = metadata["height"]
    max_px = 1024
    overlap = 0.25

    rows = 1

    if (w > max_px or h > max_px):

        max_val = max(w,h)
        #print(max_val)        
        #rows = np.ceil(max_val/final_max_px)
        #rows = (max_val - np.ceil(overlap*max_val))/(max_px - np.ceil(overlap*max_val))
        rows = np.ceil((max_val-overlap*max_px)/(max_px*(1-overlap)))

    print("rows", rows)
    print("overlap", overlap)

    # Create a vector grid for each tile
    converter.create_grid(rows, overlap, overlap)

    # Extract tiles and save
    converter.extract_tiles()


def is_raster_empty(tif_path):
    import rasterio as rio
    with rio.open(tif_path) as dataset:
        # Check the data type and dimensions
        if dataset.count == 0:
            return True  # No bands in raster

        # Read all data and check if it's entirely composed of zeros (or NoData values)
        data = dataset.read()  # Reads all bands
        if np.all(data == 0) or np.all(np.isnan(data)):
            return True  # Data contains only zeros or NaNs

    return False

def normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Normalize an image or batch of images.

    Parameters:
      image (np.ndarray): Input image array. Expected shape is either:
                          - (H, W, C) for HWC format,
                          - (C, H, W) for CHW format, or
                          - (N, C, H, W) for a batch of images.
      mean (list or tuple): Mean values for each channel.
      std (list or tuple): Standard deviation for each channel.

    Returns:
      np.ndarray: Normalized image array with the same shape as input.
    """
    image = image.astype(np.float32)/255.0
    # Reshape mean and std to broadcast along H and W dimensions
    #mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    #std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    normalized_image = (image - mean) / std

    return normalized_image

def postprocess_yolo_output(output, conf_thres=0.25, iou_thres=0.45, input_shape=(1024, 1024), orig_shape=(1024,1024)):
    #import torchvision.ops as ops  # You can replace with a custom NMS if needed

    if len(output) == 0:
        return [], []

    output = output[0]  # (1, 6, N)
    output = np.squeeze(output)  # (6, N)
    output = np.transpose(output)  # (N, 6)

    xywh = output[:, 0:4]
    objectness = output[:, 4]
    #class_conf = output[:, 5]
    scores = objectness# * class_conf

    keep = scores > conf_thres
    xywh = xywh[keep]
    scores = scores[keep]
    #classes = np.zeros_like(scores)  # YOLOv8n ONNX export may not give class index here; adjust if multiclass

    if len(xywh) == 0:
        return [], []

    # Convert xywh to xyxy
    xyxy = np.zeros_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

    # Rescale to original image size
    gain = min(input_shape[0] / orig_shape[0], input_shape[1] / orig_shape[1])
    pad_x = (input_shape[1] - orig_shape[1] * gain) / 2
    pad_y = (input_shape[0] - orig_shape[0] * gain) / 2

    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= gain

    print("xyxy shape", xyxy.shape)
    print("scores shape", scores.shape)

    return xyxy, scores

import rasterio as rio
def np2tif_2(data, filepath_tif, filepath_output, output_dtype=rio.uint8):

    # Load original tif file and copy metadata
    orig_img = rio.open(filepath_tif)
    out_meta = orig_img.meta.copy()
    out_meta.update({'count':1},indexes=1)
    out_meta.update({'dtype': output_dtype})  # Ensure output dtype is set in metadata

    # Ensure data is 2D for single-band output
    data = np.squeeze(data)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Data for single-band GeoTIFF must be 2D, got shape {data.shape}")

    # Save file
    with rio.open(filepath_output, "w", **out_meta) as dst:
        print("save file")
        dst.write(data.astype(output_dtype), 1)

# def model_inference(parameters, progress_callback = None, interruption_check = None):

#     model = parameters["model"]
#     model_dir = parameters["model_dir"]

#     results = {}

#     if model == 'HighResCanopyHeight':

#         #from .dependencies.hrch.inference_full import HRCHInference

#         model_path = os.path.join(model_dir, "HRCH_model", "HRCH_SSLhuge_satellite.onnx")

#         count = 0
#         hrch_inference = []

#         ort_sess = []
#         temp_parameters = parameters.copy() #copy parameters

#         total_tiles = len(parameters['tiles'])

#         tiles_processed = []

#         for index, tile in enumerate(parameters['tiles']):


#             prefix = os.path.basename(tile)
#             prefix, ext = os.path.splitext(prefix)
#             output_path = parameters['processed_dir']
#             processed_filepath = os.path.join(output_path, prefix + "_raster.tif")

#             if os.path.exists(processed_filepath):
#                 # # Assign results
#                 tiles_processed.append(processed_filepath)
#                 continue

#             temp_parameters.update({'input_raster_path': tile
#                                     , 'prefix': prefix
#                                     , 'output_path': output_path
#                                     })

#             if is_raster_empty(tile):
#                 continue

#             # if count == 0 and type(hrch_inference) != HRCHInference:                     
#             #     hrch_inference = HRCHInference(temp_parameters)
                
#             # hrch_inference.update(temp_parameters)
#             # hrch_inference.predict()

#             # # Assign results
#             # results["output_files"] = hrch_inference.output_files

#             #*************************************************

#             import onnxruntime as ort
#             import cv2 as cv


#             if count == 0 and type(ort_sess) != ort.InferenceSession:    

#                 providers = [
#                     ("CUDAExecutionProvider", {
#                         "device_id": 0,
#                         # Optional: additional options can be provided, e.g.
#                         #"gpu_mem_limit":  * 1024 * 1024 * 1024,
#                         #"gpu_mem_limit":  6 * 1024,
#                         # "cudnn_conv_algo_search": "EXHAUSTIVE",
#                         # "do_copy_in_default_stream": True,
#                     })
#                 ]

#                 ort_sess = ort.InferenceSession(model_path, providers=providers)
#                 #outputs = ort_sess.run(None, {'input': img.detach().numpy()})


#                 print("Available providers:", ort.get_available_providers())
#                 # Check the providers being used
#                 print("Providers in use:", ort_sess.get_providers())

#             # Load image
#             img = cv.imread(tile)
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#             img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
#             img_prec = img_prec.astype(np.float32)
#             img_prec = np.transpose(img_prec, (2,0,1))
#             img_prec = np.expand_dims(img_prec, axis=0)

#             # inference
#             outputs = ort_sess.run(None, {'input': img_prec})
#             pred = outputs[0][0]
#             pred = np.squeeze(pred)

#             # if not (np.max(pred) > 0.05):
#             #     pred = pred*0

#             pred = np.expand_dims(pred, axis=0)

#             print("**PRED SHAPE**")
#             print(pred.shape)
#             print(pred.dtype)

#             if "grayscale" in parameters["raster_outputs"]:
#                 np2tif_2(pred, tile, processed_filepath, output_dtype=rio.float32)

#             print(processed_filepath)
#             tiles_processed.append(processed_filepath)
            
#                 # #add result filepath to list of results
#                 # if not self.path_img_output.replace("_output", "_output_float") in self.output_files:
#                 #     self.output_files.append(self.path_img_output.replace("_output", "_output_float"))

            

#             # if "binary" in parameters["raster_outputs"]:
#             #     max_value = np.max(pred)
#             #     value = parameters["hrch_threshold"]*max_value
#             #     pred_binary = (pred_binary > value)*255

#             #     np2tif_2(final_img_2, self.path_img, self.path_img_output.replace("_raster","_raster_binary"))

#             #     #add result filepath to list of results
#             #     if not self.path_img_output.replace("_raster","_raster_binary") in self.output_files:
#             #         self.output_files.append(self.path_img_output.replace("_raster","_raster_binary"))

#             #*************************************************

#             if progress_callback is not None:
#                 count = index+1
#                 total = total_tiles
#                 progress = count/total
#                 status = "processing"
#                 logs = "Inference progress..."
#                 info = {
#                     "count": count
#                     , "total": total
#                     , "progress": progress
#                     , "status": status
#                     , "logs": logs
#                 }
#                 progress_callback(info)

#             if interruption_check is not None:
#                 if interruption_check():
#                     break

#             count = count + 1

#         results["tiles_processed"] = tiles_processed


#     elif model == 'DeepForest':

#         import pandas as pd
#         import geopandas as gpd


#         model_path = os.path.join(model_dir, "DeepForest.onnx")

#         count = 0
#         ort_sess = []
#         temp_parameters = parameters.copy() #copy parameters

#         total_tiles = len(parameters['tiles'])

#         tiles_processed = []

#         for index, tile in enumerate(parameters['tiles']):


#             prefix = os.path.basename(tile)
#             prefix, ext = os.path.splitext(prefix)
#             output_path = parameters['processed_dir']
#             processed_filepath = os.path.join(output_path, prefix + ".shp")

#             if os.path.exists(processed_filepath):
#                 # # Assign results
#                 tiles_processed.append(processed_filepath)
#                 continue

#             temp_parameters.update({'input_raster_path': tile
#                                     , 'prefix': prefix
#                                     , 'output_path': output_path
#                                     })

#             if is_raster_empty(tile):
#                 continue

#             import onnxruntime as ort
#             import cv2 as cv


#             if count == 0 and type(ort_sess) != ort.InferenceSession:    

#                 providers = [
#                     ("CUDAExecutionProvider", {
#                         "device_id": 0,
#                         # Optional: additional options can be provided, e.g.
#                         #"gpu_mem_limit":  * 1024 * 1024 * 1024,
#                         #"gpu_mem_limit":  6 * 1024,
#                         # "cudnn_conv_algo_search": "EXHAUSTIVE",
#                         # "do_copy_in_default_stream": True,
#                     })
#                 ]

#                 ort_sess = ort.InferenceSession(model_path, providers=providers)
#                 #outputs = ort_sess.run(None, {'input': img.detach().numpy()})


#                 print("Available providers:", ort.get_available_providers())
#                 # Check the providers being used
#                 print("Providers in use:", ort_sess.get_providers())

#             # Load image
#             img = cv.imread(tile)
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#             #img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
#             img_prec = img
#             img_prec = img_prec.astype(np.float32)/255.0
#             img_prec = np.transpose(img_prec, (2,0,1))
#             img_prec = np.expand_dims(img_prec, axis=0)

#             # inference
#             outputs = ort_sess.run(None, {'input': img_prec})
#             boxes = outputs[0] # bounding boxes
#             # output[1] # scores
#             # output[2] # labels

#             print("**BOXES SHAPE**")
#             print(boxes.shape)

#             boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

#             boxes_gdf = bb_2_geodataframe(boxes_df, temp_parameters)

#             # save shapefile
#             boxes_gdf.to_file(processed_filepath, driver='ESRI Shapefile')

#             print(processed_filepath)
#             tiles_processed.append(processed_filepath)

#             #*************************************************

#             if progress_callback is not None:
#                 count = index+1
#                 total = total_tiles
#                 progress = count/total
#                 status = "processing"
#                 logs = "Inference progress..."
#                 info = {
#                     "count": count
#                     , "total": total
#                     , "progress": progress
#                     , "status": status
#                     , "logs": logs
#                 }
#                 progress_callback(info)

#             if interruption_check is not None:
#                 if interruption_check():
#                     break

#             count = count + 1

#         results["tiles_processed"] = tiles_processed

#     elif model == "Custom ONNX Model":

#         model_path = parameters["custom_model_filepath"]

#         count = 0
#         ort_sess = []
#         temp_parameters = parameters.copy() #copy parameters

#         total_tiles = len(parameters['tiles'])

#         tiles_processed = []

#         for index, tile in enumerate(parameters['tiles']):

#             prefix = os.path.basename(tile)
#             prefix, ext = os.path.splitext(prefix)
#             output_path = parameters['processed_dir']
#             processed_filepath = os.path.join(output_path, prefix + ".shp")

#             if os.path.exists(processed_filepath):
#                 # # Assign results
#                 tiles_processed.append(processed_filepath)
#                 continue

#             temp_parameters.update({'input_raster_path': tile
#                                     , 'prefix': prefix
#                                     , 'output_path': output_path
#                                     })

#             if is_raster_empty(tile):
#                 continue

#             import onnxruntime as ort
#             import cv2 as cv
#             import pandas as pd


#             if count == 0 and type(ort_sess) != ort.InferenceSession:    

#                 providers = [
#                     ("CUDAExecutionProvider", {
#                         "device_id": 0,
#                         # Optional: additional options can be provided, e.g.
#                         #"gpu_mem_limit":  * 1024 * 1024 * 1024,
#                         #"gpu_mem_limit":  6 * 1024,
#                         # "cudnn_conv_algo_search": "EXHAUSTIVE",
#                         # "do_copy_in_default_stream": True,
#                     })
#                 ]

#                 ort_sess = ort.InferenceSession(model_path, providers=providers)
#                 #outputs = ort_sess.run(None, {'input': img.detach().numpy()})


#                 print("Available providers:", ort.get_available_providers())
#                 # Check the providers being used
#                 print("Providers in use:", ort_sess.get_providers())

#             # Load image
#             img = cv.imread(tile)
#             img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#             #img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
#             img_prec = img
#             img_prec = img_prec.astype(np.float32)/255.0
#             img_prec = np.transpose(img_prec, (2,0,1))
#             img_prec = np.expand_dims(img_prec, axis=0)

#             # inference
#             outputs = ort_sess.run(None, {'images': img_prec}) # changes for yolo
#             boxes = outputs[0][0] # bounding boxes
#             boxes = np.transpose(boxes, (1,0))
#             # output[1] # scores
#             # output[2] # labels

#             #boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.0)

#             print("**BOXES SHAPE**")
#             print(boxes.shape)

#             boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.25)

#             #boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'score'])
#             boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

#             boxes_gdf = bb_2_geodataframe(boxes_df, temp_parameters)

#             # save shapefile
#             boxes_gdf.to_file(processed_filepath, driver='ESRI Shapefile')

#             print(processed_filepath)
#             tiles_processed.append(processed_filepath)

#             #*************************************************

#             if progress_callback is not None:
#                 count = index+1
#                 total = total_tiles
#                 progress = count/total
#                 status = "processing"
#                 logs = "Inference progress..."
#                 info = {
#                     "count": count
#                     , "total": total
#                     , "progress": progress
#                     , "status": status
#                     , "logs": logs
#                 }
#                 progress_callback(info)

#             if interruption_check is not None:
#                 if interruption_check():
#                     break

#             count = count + 1

#         results["tiles_processed"] = tiles_processed

#     return results

def model_inference(parameters, progress_callback = None, interruption_check = None):

    model = parameters["model"]
    model_dir = parameters["model_dir"]

    results = {}

    output_type = ".tif"

    # Get model name
    if model == 'HighResCanopyHeight':
        model_path = os.path.join(model_dir, "HRCH_model", "HRCH_SSLhuge_satellite.onnx")
    elif model == 'DeepForest':
        model_path = os.path.join(model_dir, "DeepForestModel.onnx")
        output_type = ".shp"
    elif model == 'Mask R-CNN':
        model_path = os.path.join(model_dir, "MASKRCNNModel.onnx")
    elif model == "VHRTrees":
        model_path = os.path.join(model_dir, "VHRTrees_best.onnx")
        output_type = ".shp"
    elif model == "Custom ONNX Model":
        model_path = parameters["custom_model_filepath"]
        output_type = ".shp"
    else:
        model_path = ""


    count = 0
    ort_sess = []
    temp_parameters = parameters.copy() #copy parameters
    total_tiles = len(parameters['tiles'])
    tiles_processed = []

    for index, tile in enumerate(parameters['tiles']):

            prefix = os.path.basename(tile)
            prefix, ext = os.path.splitext(prefix)
            output_path = parameters['processed_dir']
            processed_filepath = os.path.join(output_path, prefix + output_type)

            if os.path.exists(processed_filepath):
                # # Assign results
                tiles_processed.append(processed_filepath)
                continue

            temp_parameters.update({'input_raster_path': tile
                                    , 'prefix': prefix
                                    , 'output_path': output_path
                                    })
            
            if is_raster_empty(tile):
                continue

            #************************************
            #LOAD MODEL

            import onnxruntime as ort
            import cv2 as cv
            import pandas as pd


            if count == 0 and type(ort_sess) != ort.InferenceSession:    

                providers = [
                    ("CUDAExecutionProvider", {
                        "device_id": 0,
                        # Optional: additional options can be provided, e.g.
                        #"gpu_mem_limit":  * 1024 * 1024 * 1024,
                        #"gpu_mem_limit":  6 * 1024,
                        # "cudnn_conv_algo_search": "EXHAUSTIVE",
                        # "do_copy_in_default_stream": True,
                    })
                ]

                # providers = [
                #     ("CPUExecutionProvider")
                # ]

                ort_sess = ort.InferenceSession(model_path, providers=providers)
                #outputs = ort_sess.run(None, {'input': img.detach().numpy()})


                print("Available providers:", ort.get_available_providers())
                # Check the providers being used
                print("Providers in use:", ort_sess.get_providers())

            #**************************************

            #INFERENCE DEPENDING ON THE MODEL

            # Load image
            img = cv.imread(tile)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if model == 'HighResCanopyHeight':

                
                # preprocess
                img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
                img_prec = img_prec.astype(np.float32)
                img_prec = np.transpose(img_prec, (2,0,1))
                img_prec = np.expand_dims(img_prec, axis=0)

                # inference
                outputs = ort_sess.run(None, {'input': img_prec})
                pred = outputs[0][0]
                pred = np.squeeze(pred)

                # if not (np.max(pred) > 0.05):
                #     pred = pred*0

                pred = np.expand_dims(pred, axis=0)

                print("**PRED SHAPE**")
                print(pred.shape)
                print(pred.dtype)

                if "grayscale" in parameters["raster_outputs"]:
                    np2tif_2(pred, tile, processed_filepath, output_dtype=rio.float32)
                
            elif model == 'DeepForest':

                # preprocess
                #img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
                img_prec = img
                img_prec = img_prec.astype(np.float32)/255.0
                img_prec = np.transpose(img_prec, (2,0,1))
                img_prec = np.expand_dims(img_prec, axis=0)

                # inference
                outputs = ort_sess.run(None, {'input': img_prec})
                boxes = outputs[0] # bounding boxes
                # output[1] # scores
                # output[2] # labels

                print("**BOXES SHAPE**")
                print(boxes.shape)

                boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

                boxes_gdf = bb_2_geodataframe(boxes_df, temp_parameters)

                # save shapefile
                boxes_gdf.to_file(processed_filepath, driver='ESRI Shapefile')


            elif model == 'Mask R-CNN':

                img_prec = img
                img_prec = img_prec.astype(np.float32)/255.0
                img_prec = np.transpose(img_prec, (2,0,1))
                img_prec = np.expand_dims(img_prec, axis=0)

                # inference
                outputs = ort_sess.run(None, {'input': img_prec})
                boxes = outputs[0] # bounding boxes

                masks = (outputs[3] > 0.6)*1
                masks = masks.squeeze(1)
                # Bitwise OR over the 0th axis
                merged_image = np.bitwise_or.reduce(masks)

                if "binary" in parameters["raster_outputs"]:
                    np2tif_2(merged_image, tile, processed_filepath, output_dtype=rio.uint8)

            elif model == "VHRTrees":

                # preprocess
                #img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
                img_prec = img
                img_prec = img_prec.astype(np.float32)/255.0
                img_prec = np.transpose(img_prec, (2,0,1))
                img_prec = np.expand_dims(img_prec, axis=0)

                # inference
                outputs = ort_sess.run(None, {'images': img_prec}) # changes for yolo
                boxes = outputs[0][0] # bounding boxes
                boxes = np.transpose(boxes, (1,0))
                # output[1] # scores
                # output[2] # labels

                #boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.0)

                print("**BOXES SHAPE**")
                print(boxes.shape)

                boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.25)

                #boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'score'])
                boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

                boxes_gdf = bb_2_geodataframe(boxes_df, temp_parameters)

                # save shapefile
                boxes_gdf.to_file(processed_filepath, driver='ESRI Shapefile')


            elif model == "Custom ONNX Model":


                # preprocess
                #img_prec = normalize(img, [0.420, 0.411, 0.296], [0.213, 0.156, 0.143])
                img_prec = img
                img_prec = img_prec.astype(np.float32)/255.0
                img_prec = np.transpose(img_prec, (2,0,1))
                img_prec = np.expand_dims(img_prec, axis=0)

                # inference
                outputs = ort_sess.run(None, {'images': img_prec}) # changes for yolo
                boxes = outputs[0][0] # bounding boxes
                boxes = np.transpose(boxes, (1,0))
                # output[1] # scores
                # output[2] # labels

                #boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.0)

                print("**BOXES SHAPE**")
                print(boxes.shape)

                boxes, scores = postprocess_yolo_output(outputs[0], input_shape=(960, 960), orig_shape=(960, 960), conf_thres=0.25)

                #boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax', 'score'])
                boxes_df = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

                boxes_gdf = bb_2_geodataframe(boxes_df, temp_parameters)

                # save shapefile
                boxes_gdf.to_file(processed_filepath, driver='ESRI Shapefile')

            else:
                print("No model selected")

            #**************************************

            print(processed_filepath)
            tiles_processed.append(processed_filepath)

            if progress_callback is not None:
                count = index+1
                total = total_tiles
                progress = count/total
                status = "processing"
                logs = "Inference progress..."
                info = {
                    "count": count
                    , "total": total
                    , "progress": progress
                    , "status": status
                    , "logs": logs
                }
                progress_callback(info)

            if interruption_check is not None:
                if interruption_check():
                    break

            count = count + 1

    results["tiles_processed"] = tiles_processed


    return results

def postprocess(parameters, progress_callback = None, interruption_check = None):

    # Post process
        # Generate vector outputs
        if len(parameters["vector_outputs"]) > 0:
            save_shapefile_polygon_binary_raster(parameters)
