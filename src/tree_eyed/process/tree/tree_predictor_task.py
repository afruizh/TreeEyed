# from .dependencies.utils import *
from .utils.utils_custom import *
from .dependencies.hrch import *
#from .dependencies.hrch.models import *
#from .dependencies.hrch.pl_modules import *
#import rasterio as rio
import torch
import torchvision
#import utils_custom
import os
import json
#from rasterio import merge
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import shapely.geometry
import pandas as pd
import geopandas as gpd
import cv2 as cv

from shapely.validation import make_valid



import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.io import read_image
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from deepforest import main
from deepforest import get_data
from deepforest import utilities
from collections import OrderedDict

#import sys
#sys.path.insert(0,'D:/local_mydev/project_tree/src/visualize')
#print (sys.path)

import random
import re

from .dependencies.hrch.inference_full import HRCHInference
from .custom.custom import MaskRCNNTreeInference

from ..gui_utils.config import *
from qgis.core import Qgis

from qgis.utils import iface

from qgis.core import (
  Qgis,
  QgsApplication,
  QgsMessageLog,
  QgsProcessingAlgRunnerTask,
  QgsProcessingContext,
  QgsProcessingFeedback,
  QgsProject,
  QgsTask,
  QgsTaskManager,
  QgsCoordinateReferenceSystem,
  QgsCoordinateTransform,
  QgsGeometry,
  QgsPoint
)

MODEL_NAME_MASKRCNN = "MASKRCNNModel.pth"
DEFAULT_TEMP_RASTER = "_tree_eyed_temp_raster.tif"

### function from https://discuss.pytorch.org/t/pytorch-image-segmentation-mask-polygons/87054/2
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask  


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

from qgis.core import (
  Qgis,
  QgsApplication,
  QgsMessageLog,
  QgsProcessingAlgRunnerTask,
  QgsProcessingContext,
  QgsProcessingFeedback,
  QgsProject,
  QgsTask,
  QgsTaskManager,
)

from qgis.PyQt.QtCore import pyqtSignal

import os


from .dependencies.hrch.inference_full import HRCHInference

DEFAULT_TEMP_RASTER = "_tree_eyed_temp_raster.tif"

from qgis.core import QgsMessageLog


class TreePredictorTask(QgsTask):
    
    # Additional signals
    task_finished = pyqtSignal(dict)
    
    def __init__(self, description, models_dir, parameters, np_image, extent, epsg, temp_already_saved):
        super().__init__(description, QgsTask.CanCancel)
        
        self.models_dir = models_dir
        
        self.parameters = parameters
        
        self.parameters = parameters
        self.np_image = np_image
        self.extent = extent
        self.epsg = epsg
        self.temp_already_saved = temp_already_saved


        #self.out = StringIO.StringIO()
        #sys.stdout = self.out

        QgsApplication.messageLog().messageReceived.connect(self._handle_progress)

        return
    
    def _handle_progress(self, message, tag, level):

        # with open(filename, 'a') as logfile:
        #     logfile.write('{tag}({level}): {message}'.format(tag=tag, level=level, message=message))

        if tag == "Tree Eyed Plugin":
            x = re.search("(\\d*\\.?\\d*)%",message)
            if x:
                # value = self.progress()+1
                # if value > 90:
                #     value = 90
                value = int(float(x.group(1)))
                self.setProgress(value)

    def run(self):
        
        self.setProgress(10)
        
        parameters = self.parameters
        np_image = self.np_image
        extent = self.extent
        epsg = self.epsg
        temp_already_saved = self.temp_already_saved        
        
        self.output_files = [] #Clear output files

        self.model_name = parameters['model']
        self.parameters = parameters

        print("Using model", self.model_name)

        temp_raster = os.path.join(parameters["output_path"],DEFAULT_TEMP_RASTER)

        # Save current raster
        if extent is not None and not temp_already_saved:
            #np2tif_extent(np_image, extent, epsg, DEFAULT_TEMP_OUTPUT_RASTER)
            np2tif_extent(np_image, extent, epsg, temp_raster)

        if self.model_name == 'HighResCanopyHeight':
            
            hrch_inference = HRCHInference(parameters, self.models_dir, path_img=temp_raster)
            #hrch_inference.initialize()
            hrch_inference.predict()
            self.output_files = hrch_inference.output_files

            #if not hrch_inference.img_result_binary == None:
            print("BEFORE")
            print(hrch_inference.img_result_binary.shape)
            self.output_dir = parameters["output_path"]
            self.output_prefix = parameters["prefix"]
            self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
            if len(self.parameters["vector_outputs"]) > 0:
                self.save_shapefile_polygon_binary_raster(hrch_inference.img_result_binary
                                                        , extent
                                                        , hrch_inference.img_result_binary.shape[1]
                                                        , hrch_inference.img_result_binary.shape[0]
                                                        , epsg
                                                        )

        elif self.model_name == 'Mask R-CNN':
            
            maskrcnn_tree_inference = MaskRCNNTreeInference(parameters, self.models_dir, path_img=temp_raster)
            self.output_dir = parameters["output_path"]
            self.output_prefix = parameters["prefix"]
            self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
            
            res_binary = maskrcnn_tree_inference.predict(np_image, extent = extent, epsg= epsg)
            self.output_files = maskrcnn_tree_inference.output_files

            
            if len(self.parameters["vector_outputs"]) > 0:
                self.save_shapefile_polygon_binary_raster(res_binary
                                                        , extent
                                                        , res_binary.shape[1]
                                                        , res_binary.shape[0]
                                                        , epsg
                                                        )

            # self.filepath_model = os.path.join(self.models_dir, "MASKRCNNModel.pth")
            
            # if not self.initialized:
            #     self.initialize()

            # self.output_files = [] #Clear output files
           
            # self.output_dir = parameters["output_path"]
            # self.output_prefix = parameters["prefix"]
            # self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
            # res_binary = self.predict(np_image, extent = extent, epsg= epsg)

            # self.output_dir = parameters["output_path"]
            # self.output_prefix = parameters["prefix"]
            # self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
            # if len(self.parameters["vector_outputs"]) > 0:
            #     self.save_shapefile_polygon_binary_raster(res_binary
            #                                             , extent
            #                                             , res_binary.shape[1]
            #                                             , res_binary.shape[0]
            #                                             , epsg
            #                                             )

        elif self.model_name == 'DeepForest':

            print("DeepForest")

            model = main.deepforest()
            model.create_model()
            model_path = os.path.join(self.models_dir, "NEON.pt")
            #model_path =  r"D:\local_mydata\models\trees\deepforest\NEON.pt"

            # Fix checkpoint not containing model. prefix
            checkpoint = torch.load(model_path)
            prefixed_checkpoint = OrderedDict()
            for key, value in checkpoint.items():
                prefixed_key = "model." + key  # Adjust prefix as needed
                prefixed_checkpoint[prefixed_key] = value

            model.load_state_dict(prefixed_checkpoint)

            #np_image_res = np_image.astype(np.float32)
            #np_image_res = np_image/255

            #np_image_res = model.predict_image(image=np_image_res, return_plot=True)

            temp_raster = os.path.join(parameters["output_path"],DEFAULT_TEMP_RASTER)

            boxes_df = model.predict_image(path=temp_raster, return_plot=False)

            #if not hrch_inference.img_result_binary == None:
            print("BEFORE")
            print(boxes_df)
            print(type(boxes_df))
            self.output_dir = parameters["output_path"]
            self.output_prefix = parameters["prefix"]
            self.output_filename = os.path.join(self.output_dir, self.output_prefix + "_vector.shp")
            if len(self.parameters["vector_outputs"]) > 0:
                self.save_shapefile_bb(boxes_df
                                       , extent
                                       , np_image.shape[1]
                                       , np_image.shape[0]
                                       , epsg
                                       )

            
            #print(np_image_res)
            #return np_image_res

        self.epsg
        
        self
        
        self.np_image
        
        self.results = {} 
        
        self.results["img"] = np_image
        self.results["output_files"] = self.output_files
        self.results["output_path"] = self.parameters["output_path"]
        
        print("RIGHT BEFORE")
        
        self.task_finished.emit(self.results)

        return True
    
    def finished(self, result):
        
        config_debug("finished", result)

        # if result:
        #     self.task_finished.emit(self.results)
        
        if not result:
            iface.messageBar().pushMessage("Error", "Inference task could not be completed", level=Qgis.Critical)
        
        return

    
    def save_shapefile(self,results, extent, img_width, img_height, epsg):

        df_tree_polygons_test = pd.DataFrame()
        tree_bb = []

        count = 0

        


        #Postprocessing
        bb = (results["boxes"])
        # Apply non maximum suppresion
        res = torchvision.ops.nms(results["boxes"], results["scores"], iou_threshold = 0.98)
        #pred_boxes = bb.long()
        pred_boxes = bb.long()[res].long()

        for bbox in pred_boxes:

            print("detection", count)
            print(bbox)

            #bbox = detection["bbox"]
            # mask = detection["mask"]
            # score = detection["score"]
            # category_id = detection["category_id"]

            # Convert bounding box to COCO format
            x_min, y_min, x_max, y_max = bbox
            #x_max = x_min + w
            #y_max = y_min + h
            #bbox = [x_min, y_min, x_max, y_max]

            xtl = x_min
            ytl = y_min
            width = abs(x_max-x_min)
            height = abs(y_max-y_min)
            

            bb = [self.pos2coords((xtl, ytl), extent, img_width, img_height)
            ,self.pos2coords((xtl + width, ytl), extent, img_width, img_height)
            ,self.pos2coords((xtl + width, ytl + height), extent, img_width, img_height)
            ,self.pos2coords((xtl, ytl + height), extent, img_width, img_height)]

            bb = shapely.geometry.Polygon(bb)

            df_item_test = pd.DataFrame({'Class': 'Tree'
                                        , 'ID': count
                                        , 'label': 'Tree'
                                        }
                                        , index=[count])
            df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))

            tree_bb.append(bb)

            count = count + 1

        # ## test bounding box

        # bb0 = [(extent.xMinimum(), extent.yMinimum())
        #       , (extent.xMaximum(), extent.yMinimum())
        #       , (extent.xMaximum(), extent.yMaximum())
        #       , (extent.xMinimum(), extent.yMaximum())]
            
        # print("bb0", bb0)
            
        # xtl = 0
        # ytl = 0
        # width = img_width
        # height = img_height

        
        
        # bb = [self.pos2coords((xtl, ytl), extent, img_width, img_height)
        #     ,self.pos2coords((xtl + width, ytl), extent, img_width, img_height)
        #     ,self.pos2coords((xtl + width, ytl + height), extent, img_width, img_height)
        #     ,self.pos2coords((xtl, ytl + height), extent, img_width, img_height)]
        
        # print("bb0", bb)

        # bb = shapely.geometry.Polygon(bb)

        # df_item_test = pd.DataFrame({'Class': 'Tree'
        #                             , 'ID': count
        #                             , 'label': 'Tree'
        #                             }
        #                             , index=[count])
        # df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))
        # tree_bb.append(bb)

        # # #*******************

        gdf_trees = gpd.GeoDataFrame(df_tree_polygons_test, geometry=tree_bb)
        gdf_trees = gdf_trees.set_crs(epsg=epsg.replace("EPSG:",""))

        gdf_trees.to_file(DEFAULT_TEMP_OUTPUT_SHP, index=False)

    def save_shapefile_polygon_binary_raster(self, thresh, extent, img_width, img_height, epsg):

        #if 3 channels, convert to single channel
        if len(thresh.shape)==3 and thresh.shape[2]==3:
            thresh = cv.cvtColor(thresh, cv.COLOR_RGB2GRAY)
            

        df_tree_polygons_test = pd.DataFrame()
        tree_bb = []

        (contours, hierarchy) = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        count = 0
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
                new_coord = self.pos2coords(coord, extent, img_width, img_height)
                coord_polygon.append(new_coord)

            
            if len(coord_polygon) > 2:#at least 3 points

                polygon_object = shapely.geometry.Polygon(coord_polygon)
                polygon_object = make_valid(polygon_object)

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
        
        config_debug("len", len(gdf_trees))

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

        # Add additional

        lon = extent.xMinimum()
        lat = extent.yMinimum()

        # Guarantee correct lat lon crs
        #source_crs = layer.crs()
        source_crs = QgsCoordinateReferenceSystem(epsg)
        target_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

        geom = QgsGeometry(QgsPoint(extent.xMinimum(), extent.yMinimum()))
        geom.transform(transform)

        lon = geom.constGet().x()
        lat = geom.constGet().y()

        new_crs = "+proj=cea +lat_0=" + str(lat)  + " +lon_0="+ str(lon) + " +units=m"
        #print(new_crs)
        #df = df.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")
        gdf_trees["area_m2"] = gdf_trees.to_crs(new_crs).area
        gdf_trees["perim_m"] = gdf_trees.to_crs(new_crs).length
        gdf_trees["a_diam_m"] = np.sqrt(gdf_trees["area_m2"]*4.0/np.pi)
        
        config_debug("****************COLS")
        config_debug(gdf_trees.columns)

        #save
        if "polygons" in self.parameters["vector_outputs"]:
            config_debug(gdf_trees.columns)
            config_debug(self.output_filename)
            gdf_trees.to_file(self.output_filename, index=False)
            if not self.output_filename in self.output_files:
                self.output_files.append(self.output_filename)

        if "bounding_boxes" in self.parameters["vector_outputs"]:
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
            new_output_filename = self.output_filename.replace("_vector.shp", "_vector_bb.shp")
            gdf_trees_bb.to_file(new_output_filename, index=False)
            if not new_output_filename in self.output_files:
                self.output_files.append(new_output_filename)

        if "centroids" in self.parameters["vector_outputs"]:
            gdf_trees_c = gdf_trees.copy()
            gdf_trees_c['geometry'] = gdf_trees_c['geometry'].centroid
            new_output_filename = self.output_filename.replace("_vector.shp", "_vector_centroids.shp")
            gdf_trees_c.to_file(new_output_filename, index=False)
            if not new_output_filename in self.output_files:
                self.output_files.append(new_output_filename)

    def save_shapefile_bb(self, df, extent, img_width, img_height, epsg):

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
                new_coord = self.pos2coords(coord, extent, img_width, img_height)
                coord_polygon.append(new_coord)

            print("len", len(coord_polygon))

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
        gdf_trees = gdf_trees.set_crs(epsg=epsg.replace("EPSG:",""))

        print(gdf_trees)

        # Dissolve so no items in others
        #gdf_trees = gdf_trees.dissolve()
        #gdf_trees = gdf_trees.explode()

        
        # Add additional

        lon = extent.xMinimum()
        lat = extent.yMinimum()

        # Guarantee correct lat lon crs
        #source_crs = layer.crs()
        source_crs = QgsCoordinateReferenceSystem(epsg)
        target_crs = QgsCoordinateReferenceSystem("EPSG:4326")

        transform = QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

        geom = QgsGeometry(QgsPoint(extent.xMinimum(), extent.yMinimum()))
        geom.transform(transform)

        lon = geom.constGet().x()
        lat = geom.constGet().y()

        new_crs = "+proj=cea +lat_0=" + str(lat)  + " +lon_0="+ str(lon) + " +units=m"
        #print(new_crs)
        #df = df.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")
        gdf_trees["area_m2"] = gdf_trees.to_crs(new_crs).area
        gdf_trees["a_diam_m"] = np.sqrt(gdf_trees["area_m2"]*4.0/np.pi)

        #save

        if "bounding_boxes" in self.parameters["vector_outputs"]:
            gdf_trees_bb = gdf_trees.copy()
 
            new_output_filename = self.output_filename.replace("_vector.shp", "_vector_bb.shp")
            gdf_trees_bb.to_file(new_output_filename, index=False)
            if not new_output_filename in self.output_files:
                self.output_files.append(new_output_filename)

        if "centroids" in self.parameters["vector_outputs"]:
            gdf_trees_c = gdf_trees.copy()
            gdf_trees_c['geometry'] = gdf_trees_c['geometry'].centroid
            new_output_filename = self.output_filename.replace("_vector.shp", "_vector_centroids.shp")
            gdf_trees_c.to_file(new_output_filename, index=False)
            if not new_output_filename in self.output_files:
                self.output_files.append(new_output_filename)

    

    def add_zonal_statistics(self, raster_path):

        return


    
    def save_shapefile_polygon(self,results, extent, img_width, img_height, epsg):

        df_tree_polygons_test = pd.DataFrame()
        tree_bb = []

        count = 0



        #Postprocessing
        #polygons = (results["masks"])
        polygons = (results["masks"] > 0.7).squeeze(1)

        bb = (results["boxes"])
        # Apply non maximum suppresion
        res = torchvision.ops.nms(results["boxes"], results["scores"], iou_threshold = 0.4)
        #pred_boxes = bb.long()



        pred_polygons = polygons.long()[res].long()
        print("len", len(pred_polygons))
        #pred_polygons = polygons.long().long()

        for polygon in pred_polygons:

            #print("detection", count)
            #print(polygon)

            # Convert mask to polygon
            polygon_np = polygon.squeeze().detach().cpu().numpy()
            #print(np.max(polygon_np))
            polygon_np = random_colour_masks(polygon_np)
            gray_mask = cv.cvtColor(polygon_np, cv.COLOR_RGB2GRAY)
            ret, thresh = cv.threshold(gray_mask, 127, 255, 0)

            # add parenthesis for some reason to solve the problem??
            (contours, hierarchy) = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            #print("count", count)
            #print(contours)

            #window_name = "mask"

            # if count < 4:
            #     h = thresh.shape[0]
            #     w = thresh.shape[1]
            #     cv.namedWindow(window_name, cv.WINDOW_NORMAL)
            #     cv.resizeWindow(window_name, w, h)
            #     cv.imshow(window_name, thresh)
            #     cv.waitKey(1)

            #print(contours)
            #print(type(contours))

            #opencv version contours return points and centroid?
            #new_contours = 
            #for contour in contours 


            #contours = np.squeeze(contours)
            contours = np.squeeze(contours)

            coord_polygon =[]
            for point in contours:
                #print("here")
                #print(point)
                
                coord = (point[0], point[1])
                new_coord = self.pos2coords(coord, extent, img_width, img_height)

                coord_polygon.append(new_coord)

            if len(coord_polygon) > 0:

                polygon_object = shapely.geometry.Polygon(coord_polygon)

                df_item_test = pd.DataFrame({'Class': 'Tree'
                                            , 'ID': count
                                            , 'label': 'Tree'
                                            }
                                            , index=[count])
                df_tree_polygons_test = pd.concat((df_tree_polygons_test, df_item_test))

                tree_bb.append(polygon_object)

            count = count + 1

        gdf_trees = gpd.GeoDataFrame(df_tree_polygons_test, geometry=tree_bb)
        gdf_trees = gdf_trees.set_crs(epsg=epsg.replace("EPSG:",""))

        #gdf_trees.to_file("D:/local_mydata/tree/results/vector/result_polygons.shp")

        #save
        gdf_trees.to_file(self.output_filename, index=False)
        if not self.output_filename in self.output_files:
            self.output_files.append(self.output_filename)


    def save_coco_results(self, image_paths, results, output_path):
        """
        Save Mask-RCNN results in COCO format.

        Args:
            image_ids: List of image IDs.
            results: List of dictionaries containing predicted bounding boxes, masks, and scores.
            coco_dict: COCO dictionary containing information about the dataset.
            output_path: Path to save the COCO JSON file.
        """

        coco_dict = {
        "dataset": {
            "name": "MyDataset",
            "description": "A dataset of images with objects",
            "license": "CC BY-SA 4.0",
        },
        "categories": [
            {"id": 1, "name": "tree", "supercategory": ""}
        ],
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 256, "height": 256},
        ],
        }

        images = []
        image_ids = []
        for i, image_path in enumerate(image_paths):
            images.append({
                "id": 1
                , "file_name": image_path
                , "width": 256
                , "height": 256
            })

        coco_anns = []
        for i, image_id in enumerate(image_ids):
            for detection in results[i]:

                #Postprocessing
                bb = (detection["boxes"])
                # Apply non maximum suppresion
                res = torchvision.ops.nms(detection["boxes"], detection["scores"], iou_threshold = 0.2)
                #pred_boxes = bb.long()
                pred_boxes = bb.long()[res].long()


                bbox = detection["bbox"]
                mask = detection["mask"]
                score = detection["score"]
                category_id = detection["category_id"]

                # Convert bounding box to COCO format
                x_min, y_min, w, h = bbox
                x_max = x_min + w
                y_max = y_min + h
                bbox = [x_min, y_min, x_max, y_max]

                # Convert mask to COCO format
                #encoded_mask = encode(np.asfortranarray(mask)).tolist()

                # Create COCO annotation
                ann = {
                    "image_id": image_id,
                    "category_id": category_id,
                    #"segmentation": [encoded_mask],
                    "area": mask.sum().item(),
                    "bbox": bbox,
                    "score": score,
                }
                coco_anns.append(ann)

        # Create COCO object and save results
        coco = COCO()
        coco.dataset = coco_dict["dataset"]
        coco.cats = coco_dict["categories"]
        coco.imgs = coco_dict["images"]
        coco.anns = coco_anns

        with open(output_path, "w") as f:
            json.dump(coco, f)

    # def coords2pos(self, tile_grid, coord, pixel_w, pixel_h):

 
    #     xmin, ymin, xmax, ymax = tile_grid.total_bounds

    #     width = abs(xmax - xmin)
    #     height = abs(ymax - ymin)

    #     x = (coord[0] - xmin)/width
    #     y = 1.0 - (coord[1] - ymin)/height

    #     #res = (round(x*pixel_w, 2),round(y*pixel_h, 2))

    #     return (x*pixel_w, y*pixel_h)
    
    def pos2coords(self, pos, extent, img_width, img_height):

        # print(extent)
        # print(extent.xMinimum())
        # print(extent.xMaximum())
        # print(extent.yMinimum())
        # print(extent.yMaximum())
        # print(extent.width())
        # print(extent.height())
        # print(self.iface.mapCanvas().mapSettings().destinationCrs().authid())

        coord_x_min = extent.xMinimum()
        coord_y_min = extent.yMinimum()

        coord_width = extent.width()
        coord_height = extent.height()
 
        x = (pos[0])/img_width
        y = 1.0 - (pos[1])/img_height

        coord_x = x*coord_width + coord_x_min
        coord_y = y*coord_height + coord_y_min

        #res = (round(x*pixel_w, 2),round(y*pixel_h, 2))

        return (coord_x, coord_y)
