from rasterio.mask import mask
import rasterio as rio
#import numpy as np
import os
#import subprocess
import cv2 as cv
#import json
#from PIL import Image
import geopandas as gpd

import shapely



'''
HANDLING FILES
'''

def get_files(dir, ext = 'dng'):
    """
    Get files with ext in directory
    """
    files = os.listdir(path = dir)
    files = [f for f in files if f.endswith('.' + ext)]
    basenames = [os.path.splitext(f)[0] for f in files] #get only file basename

    return files, basenames

'''
NUMPY ARRAY TO TIFF
'''

def np2tif(data, filepath_tif, filepath_output):
    """save numpy array to tif
    """

    # Load original tif file and copy metadata
    orig_img = rio.open(filepath_tif)
    out_meta = orig_img.meta.copy()
    out_meta.update({'count':1})
    output_dtype = rio.uint8

    # Save file
    with rio.open(filepath_output, "w", **out_meta) as dst:
        dst.write(data.astype(output_dtype),indexes=1)


def np2tif_2(data, filepath_tif, filepath_output):

    # Load original tif file and copy metadata
    orig_img = rio.open(filepath_tif)
    out_meta = orig_img.meta.copy()
    out_meta.update({'count':1})
    output_dtype = rio.uint8

    # Save file
    with rio.open(filepath_output, "w", **out_meta) as dst:
        dst.write(data.astype(output_dtype))


def np2tif_3(data, filepath_tif, filepath_output):
    """save numpy array to tif multiband
    """

    # Load original tif file and copy metadata
    orig_img = rio.open(filepath_tif)
    out_meta = orig_img.meta.copy()
    out_meta.update({'count':3})
    output_dtype = rio.uint8

    # Save file
    with rio.open(filepath_output, "w", **out_meta) as dst:
        dst.write(data.astype(output_dtype))

def np2tif_extent(np_image, extent, epsg, filepath):
    """save numpy to tif by extent and epsg
    """

    (h,w,channels) = np_image.shape

    np_image = cv.cvtColor(np_image, cv.COLOR_BGR2RGB)

    np_image = np_image.transpose((2, 0, 1))
    print(np_image.shape)

    print(extent)
    print(np_image.dtype)

    profile = {'driver': 'GTiff'
               , 'height': h
               , 'width': w
               , 'count': channels
               , 'dtype': rio.uint8
               , 'transform': rio.transform.from_bounds(extent.xMinimum()
                                                        , extent.yMinimum()
                                                        , extent.xMaximum()
                                                       , extent.yMaximum()
                                                        , w, h)
               }

    print(profile)

    with rio.open(filepath, 'w', crs=epsg, **profile) as dst:
        dst.write(np_image)

    #with rio.open(filepath, 'w', crs='EPSG:3857', **profile) as dst:
    #    pass # write data to this Web Mercator projection dataset.


def raster_extract(raster_filepath, extent, epsg, filepath):
    """extract raster region by extent and epsg
    """
    

    with rio.open(raster_filepath) as src:

        geom = shapely.geometry.box(extent.xMinimum()
                   , extent.yMinimum()
                   , extent.xMaximum()
                   , extent.yMaximum())
        
        gdf = gpd.GeoDataFrame(geometry=[geom])
        gdf = gdf.set_crs(epsg=epsg.replace("EPSG:",""))
        gdf = gdf.to_crs(crs=src.crs)
        geom = gdf.geometry[0]
                               
                               
        
        #src =rio.reproject(rio.crs.CRS.from_string(epsg))


        


        out_img, out_transform = rio.mask.mask(src, [geom], crop=True)

        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": out_img.shape[1],
                        "width": out_img.shape[2],
                        "transform": out_transform})

        with rio.open(filepath,"w", **out_meta) as dest:
            dest.write(out_img)


    return