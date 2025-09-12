import os
# Additional imports
import qgis
#from qgis.core import QgsCoordinateTransform
from qgis.core import QgsRasterLayer
#from qgis.core import QgsRectangle
from qgis.core import QgsVectorLayer
from qgis.core import QgsRasterLayer
#from qgis.gui import QgsMapCanvas
#from qgis.gui import QgsMapLayerProxyModel
from qgis.core import QgsMapSettings
from qgis.core import QgsMapRendererCustomPainterJob

from qgis.core import QgsRasterRendererUtils
from qgis.core import QgsSingleBandPseudoColorRenderer
from qgis.core import QgsRasterShader 
from qgis.core import QgsColorRampShader

from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QImage, QPainter, QColor

#from qgis.PyQt.QtWidgets import QProgressDialog
from qgis.PyQt.QtWidgets import QMessageBox, QFileDialog


from qgis.core import QgsProject

from qgis.core import (
  QgsSettings
  , QgsTask
#  , QgsTaskManager
  , QgsApplication
  , QgsMessageLog
  , QgsStyle
)

from qgis.core import Qgis

#from qgis.analysis import QgsZonalStatistics

from qgis.core import QgsCoordinateTransform
from qgis.core import QgsCoordinateReferenceSystem

from .process.gui_utils.config import *
from .process.gui_utils.qgis_utils import *

#from .process.tree.metrics.compare import COCOMetrics

from time import sleep
import random
MESSAGE_CATEGORY = 'Tree Eyed Plugin'

# from .process.tree.utils.utils_custom import *
# from .process.tree.qgis2coco.qgis2coco import *

from pathlib import Path

#from .process.gui_utils.installer import InstallerManager

import cv2 as cv
import numpy as np
import glob

#from .process.tree.interface.processor import Processor
from .process.tree.interface.processor import Processor

DEFAULT_RASTER_COLORMAP = "colormap.txt"
DEFAULT_VECTOR_BB = "detection_style.qml"
DEFAULT_TEMP_RASTER = "_tree_eyed_temp_raster.tif"

from qgis.PyQt.QtCore import pyqtSignal

class WorkerTask(QgsTask):

    # Additional signals
    task_finished = pyqtSignal(dict)

    def __init__(self, description, params):
        super().__init__(description, QgsTask.CanCancel)
        self.params = params

    def _handle_progress(self, info):
        self.setProgress(info["progress"]*90) # use 90 to avoid reaching 100% before the end

    def run(self):
        """Long-running task."""

        processor = Processor(self.params
                              , progress_callback = self._handle_progress
                              , interruption_check = self.isCanceled)
        results = processor.run()

        #QgsMessageLog.logMessage("Inference process started", MESSAGE_CATEGORY, Qgis.Warning)

        print("***RESULT")
        print(results)

        if results["status"] == "error":

            QgsMessageLog.logMessage(results["status"]+ " " + results["message"],MESSAGE_CATEGORY, Qgis.Critical)
            return False
        

        self.task_finished.emit(results)

        self.results = results

        return True

    def finished(self, result):

        if not result:
            iface.messageBar().pushMessage("Error", "Task could not be completed", level=Qgis.Critical)
        else:
            #iface.messageBar().pushMessage("Finished", self.description() + " completed", level=Qgis.Success)
            iface.messageBar().pushMessage(self.results["status"].capitalize(), self.description() + ": " + self.results["log"], level=Qgis.Success)

        return

class TreeEyedProcessor:

    def __init__(self, iface):

        self.iface = iface

        self.task_manager = QgsApplication.taskManager() # Solve bug not running first time?

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # update default paths
        global DEFAULT_RASTER_COLORMAP
        DEFAULT_RASTER_COLORMAP = os.path.join(self.plugin_dir, "colormap.txt")
        global DEFAULT_VECTOR_BB
        DEFAULT_VECTOR_BB = os.path.join(self.plugin_dir, "detection_style.qml")
        
        # Read existing settings
        model_dir = self._get_models_dir()
        
        if model_dir == "NODATA":            
            self._prompt_download_models()

    def _check_already_exist(self, parameters):
        """checks if output settings are valid

        Args:
            parameters (dict): contains the dict with the processing parameters

        Returns:
            Bool: True if already exists otherwise False
        """
        
        root_dir = parameters['output_path']
        pattern = os.path.join(root_dir, parameters["prefix"] + "_*")
        print(pattern)
        files = glob.glob(pattern)

        if len(files) > 0:
            return True
        
        return False
    
    def _process_filter_area(self, parameters):
        """filter by area process

        Args:
            parameters (dict): contains the dict with the processing parameters
        """
        
        # area_value = parameters["filter_area_area"]
        
        
        # selected_layer = parameters["filter_area_layer"]
        # layer_path = selected_layer.dataProvider().dataSourceUri()
        
        # gdf = gpd.read_file(layer_path)       
        # gdf = gdf[gdf["area_m2"] <= area_value]
        
        # output_dir = parameters["output_path"]
        # output_prefix = parameters["prefix"]
        # output_filename = os.path.join(output_dir, output_prefix + "_vector.shp")
        
        
        # gdf.to_file(output_filename, index=False)
        
        # layers = self._add_processed_layers([output_filename])
        
        # return

        selected_layer = parameters["filter_area_layer"]
        layer_path = selected_layer.dataProvider().dataSourceUri()

        parameters['input_raster_path'] = layer_path

        qgstask = WorkerTask("Filter by area", parameters)
        qgstask.task_finished.connect(self._process_task_finished)
        QgsApplication.taskManager().addTask(qgstask)  
    
    def _process_raster2vector(self, parameters):
        """convert raster 2 vector

        Args:
            parameters (dict): contains the dict with the processing parameters
        """
        
        ## Fix result types 
        parameters["vector_outputs"] = ["polygons"]
        parameters["raster_outputs"] = []
        

        #selected_layer = parameters['layer']
        selected_layer = parameters["raster2vector_layer"]
        layer_tree_root = QgsProject.instance().layerTreeRoot()

        layer = layer_tree_root.findLayer(selected_layer.id())

        config_debug(selected_layer.id())
        layer_path = selected_layer.dataProvider().dataSourceUri()
        config_debug(layer_path)

        extent = selected_layer.extent()
        config_debug(extent)

        #return

        #img = cv.imread(layer_path, cv.IMREAD_GRAYSCALE)
        #config_debug(img.dtype)
        #config_debug(img.shape)
        #config_debug("max",np.max(img))
        
        #percentage = parameters["raster2vector_threshold"]/100.0
        
        #max_value = np.max(img)
        #value = percentage*max_value
        #ret, img = cv.threshold(img, value, 255, 0)
        
        #config_debug("max",np.max(img))
        
        # #Visualize
        # window_name = "Inference"
        # h = img.shape[0]
        # w = img.shape[1]
        # #print(img_bgr.shape)
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_name, w, h)
        # cv.imshow(window_name, img)
        # cv.waitKey(1)
        # return

        parameters["binary_raster_path"] = layer_path
        
        

        # extent = self.iface.mapCanvas().extent()
        #epsg = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

        # img = self._capture_canvas(parameters["layer"], visible=True)
        #model_dir = self._get_models_dir()


        parameters['task'] = "raster2vector"

        qgstask = WorkerTask("Raster to vector task", parameters)
        qgstask.task_finished.connect(self._process_task_finished)
        QgsApplication.taskManager().addTask(qgstask)  


        # from .process.tree.tree_predictor_task import TreePredictorTask
        # tree_predictor_task = TreePredictorTask("Tree predictor task", model_dir, parameters, img, extent, epsg, temp_already_saved = False)

        # tree_predictor_task.output_dir = parameters["output_path"]
        # tree_predictor_task.output_prefix = parameters["prefix"]
        # tree_predictor_task.output_filename = os.path.join(tree_predictor_task.output_dir, tree_predictor_task.output_prefix + "_vector.shp")
               
        # tree_predictor_task.output_files = []
        # #tree_predictor_task.output_files.append(tree_predictor_task.output_filename)
        
        

        # tree_predictor_task.save_shapefile_polygon_binary_raster(img
        #                                                , extent
        #                                                , img.shape[1]
        #                                                , img.shape[0]
        #                                             , epsg
        #                                                )
        
        # layers = self._add_processed_layers(tree_predictor_task.output_files)


        # # # Calculate zonal statistics
        # # # Create zonal statistics object
        # # zonal_stats = QgsZonalStatistics(layers[0], selected_layer
        # #                                 , attributePrefix="height"
        # #                                 , rasterBand=1, stats=QgsZonalStatistics.Statistics(QgsZonalStatistics.Max)
        # #                                 )

        # # # Configure statistics
        # # #zonal_stats.setStatistics(QgsZonalStatistics.Mean)

        # # # Calculate zonal statistics
        # # zonal_stats.calculateStatistics(None)
        # # #zonal_stats.calculateStatistics(QgsZonalStatistics.SecondPass)

        # # # for field in layer.fields():
        # # # if field.name() == 'old_fieldname':

        # # #     with edit(layer):
        # # #         idx = layer.fields().indexFromName(field.name())
        # # #         layer.renameAttribute(idx, 'new_fieldname')



    def _process_validate(self, parameters):
        """calculates validation metrics betwee 2 COCO datasets in .json format

        Args:
            parameters (dict): contains the dict with the processing parameters
        """
        
        validate_ground_truth = parameters["validate_ground_truth"]
        validate_prediction = parameters["validate_prediction"]
        
        if os.path.exists(validate_ground_truth) and os.path.exists(validate_prediction):
        
            from .process.tree.metrics.compare import COCOMetrics
            coco_metrics = COCOMetrics()
            coco_metrics.load_target(validate_ground_truth, result_type='coco')
            coco_metrics.load_pred(validate_prediction, result_type='coco')
            coco_metrics.compute()
            
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText(coco_metrics.final_message)
            msg.setIcon(QMessageBox.Information)
            msg.show()
            
        else:
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Selected files are invalid")
            msg.setIcon(QMessageBox.Critical)
            msg.show()

    def _add_processed_layers(self, output_files):
        """add the resulting layers to the current project

        Args:
            output_files (string list): list of layer files 

        Returns:
            layer: layers added to the project
        """

        #print("output_files")
        #print(self.predictor.output_files)

        layers = []
            
        #for file in self.predictor.output_files:
        for file in output_files:
            
            print(file)
            name_stem = "results"
            name_stem = Path(file).stem
            print("name_stem", name_stem)
            if ".tif" in file:
                layer = QgsRasterLayer(file, name_stem)
                
                

                # config_debug("max_value", max_value)
                

                # #https://docs.qgis.org/3.34/en/docs/pyqgis_developer_cookbook/raster.html
                # fcn = QgsColorRampShader()
                # fcn.setColorRampType(QgsColorRampShader.Interpolated)
                # color_ramp_load = QgsRasterRendererUtils.parseColorMapFile(DEFAULT_RASTER_COLORMAP)
                # #lst = [ QgsColorRampShader.ColorRampItem(0, QColor(0,255,0)),
                # #    QgsColorRampShader.ColorRampItem(255, QColor(255,255,0))]
                # lst0 = color_ramp_load[1]
                
                # color0 = lst0[0].color
                # color1 = lst0[1].color
                # config_debug("colors")
                # config_debug(color0.red(), color0.green(), color0.blue(), color0.alpha())
                # config_debug(color1.red(), color1.green(), color1.blue(), color1.alpha())
                
                # color = QColor(19,222,222,255)
                
                # lst = []
                # lst.append(lst0[0])
                # lst.append(QgsColorRampShader.ColorRampItem(max_value,color,lst0[1].label))

                # fcn.setColorRampItemList(lst)
                # shader = QgsRasterShader()
                # shader.setRasterShaderFunction(fcn)                
                
                # renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
                # layer.setRenderer(renderer)


                if not layer.isValid():
                    print("Layer failed to load!", )
                else:
                    QgsProject.instance().addMapLayer(layer)
                    layers.append(layer)

                    t_image = cv.imread(file,cv.IMREAD_UNCHANGED)
                    min_value = np.min(t_image)
                    max_value = np.max(t_image)


                    # Add color ramp to layer using default color ramp
                    rocket_ramp = QgsStyle().defaultStyle().colorRamp('Viridis')

                    # Sample colors at intervals (for smooth ramp, use more steps)
                    steps = 50
                    items = []
                    for i in range(steps + 1):
                        value = min_value + (max_value - min_value) * i / steps
                        color = rocket_ramp.color(i / steps)
                        if i == 0:
                            color.setAlpha(0)
                        color_item = QgsColorRampShader.ColorRampItem(value, color, str(value))
                        items.append(color_item)

                    shader_func = QgsColorRampShader()
                    shader_func.setColorRampType(QgsColorRampShader.Interpolated)
                    shader_func.setColorRampItemList(items)

                    shader = QgsRasterShader()
                    shader.setRasterShaderFunction(shader_func)

                    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
                    renderer.setClassificationMin(min_value)
                    renderer.setClassificationMax(max_value)
                    layer.setRenderer(renderer)
                    layer.triggerRepaint()


            elif ".shp" in file:
                #layer = QgsVectorLayer(file, "results", "ogr")
                layer = QgsVectorLayer(file, name_stem)

                

                if not layer.isValid():
                    print("Layer failed to load!")
                else:
                    
                    if "_bb.shp" in file:
                        print("DEFAULT_VECTOR_BB", DEFAULT_VECTOR_BB)
                        layer.loadNamedStyle(DEFAULT_VECTOR_BB)
                        layer.triggerRepaint()

                    QgsProject.instance().addMapLayer(layer)

                    
                    layers.append(layer)

                    

        return layers

    def _process(self, parameters, is_task=True):
        """main processing, checks, task selection and calling

        Args:
            parameters (dict): contains the dict with the processing parameters
            is_task (bool, optional): If the processing is a QGIS task. Defaults to True.
        """

        global DEFAULT_RASTER_COLORMAP

        config_debug('processing...')
        config_debug('parameters', parameters)
        
        if not self._check_current_tasks():
            return
        
        if parameters['task'] == "validate":
            self._process_validate(parameters)
            return

        extent_type = parameters["extent_type"]
        
        # Show warning message if output directory is empty
        if (parameters['output_path'] == ''):
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Please select an output directory")
            msg.setIcon(QMessageBox.Information)
            msg.show()
            return
        
        # Show warning message if outputs already exists
        if (self._check_already_exist(parameters)):
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Output name already exists in output directory. Please select a different output name.")
            msg.setIcon(QMessageBox.Information)
            msg.show()
            return
        
        # Check if model files exist
        if not self._check_valid_model(parameters):
            return

        # CHeck Results types are selected
        if parameters['task'] == "inference" and not self._check_result_types(parameters):
            return

        # If task is raster2vector
        if parameters['task'] == "raster2vector":
            self._process_raster2vector(parameters)            
            return
        
        # If task is filter_area
        if parameters['task'] == "filter_area":
            self._process_filter_area(parameters)            
            return
        
        # If task is capture
        if parameters['task'] == "capture":
            #print("NOT IMPLEMENTED")

            img = self._capture_canvas(parameters["layer"], visible=True, white_background = False)

            img_bgr = img

            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

            extent = self.iface.mapCanvas().extent()
            epgs = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

            capture_raster = os.path.join(parameters["output_path"],"capture_raster.tif")

            parameters["is_temporal"] = True

            # Save current raster
            if extent is not None:
                #np2tif_extent(np_image, extent, epsg, DEFAULT_TEMP_OUTPUT_RASTER)
                np2tif_extent(img, extent, epgs, capture_raster)


            return
        
        if parameters['task'] == "export_dataset":

            image_path = parameters["input_image"].dataProvider().dataSourceUri()
            annotations_path = parameters["annotations"].dataProvider().dataSourceUri()
            num_tiles = parameters["num_tiles"] # now it would be max pixels per tile
            overlap = int(parameters["overlap"])/100.0
            output_format = "." + parameters["output_format"].lower()
            
            dir_name = parameters["prefix"] + "_coco_dataset"
            
            path_output = os.path.join(parameters["output_path"], dir_name)


            parameters["image_path"] = image_path
            parameters["annotations_path"] = annotations_path


            #Check if already exist
            if os.path.exists(path_output):
                msg = QMessageBox(self.iface.mainWindow())
                msg.setWindowTitle("Tree Eyed")
                msg.setText("Output directory {} already exists. Please select a different output name.".format(dir_name))
                msg.setIcon(QMessageBox.Information)
                msg.show()
                return

            qgstask = WorkerTask("Export dataset task", parameters)
            QgsApplication.taskManager().addTask(qgstask)            

            # #from .process.tree.qgis2coco.qgis2coco import QGIS2COCO
            # from .process.tree.qgis2coco.qgis2coco import QGIS2COCO
            # from .process.tree.qgis2coco.qgis2coco import check_raster

            # print("HEHEHE")
            # print(image_path)

            # metadata_final = check_raster(image_path)
            # w = metadata_final["width"]
            # h = metadata_final["height"]

            # max_px = num_tiles

            # rows = 1

            # if (w > max_px or h > max_px):

            #     max_val = max(metadata_final["width"], metadata_final["height"])
            #     rows = np.ceil((max_val-overlap*max_px)/(max_px*(1-overlap)))

            # COCO_CONTRIBUTOR = "TreeEyed Plugin | Tropical Forages Program | Alliance Bioversity International & CIAT"
            # COCO_LICENSE = "Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"
            # COCO_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
            # COCO_INFORMATION = ""

            # exporter = QGIS2COCO(image_path
            #         , annotations_path
            #         , allow_clipped_annotations = False
            #         , allow_no_annotations = False
            #         , information = COCO_INFORMATION
            #         , license = COCO_LICENSE
            #         , license_url = COCO_LICENSE_URL
            #         , contributor = COCO_CONTRIBUTOR
            #         , output_format = output_format
            #         )
            # exporter.convert(path_output, rows = rows, overlap = overlap)

            return
        
        # Show warning message custom extent for WMS layer
        if extent_type == "Custom extent":

            layer = parameters["layer"]

            if layer.providerType() == 'wms':
                msg = QMessageBox(self.iface.mainWindow())
                msg.setWindowTitle("Tree Eyed")
                msg.setText("Option not available for WMS layers")
                msg.setIcon(QMessageBox.Warning)
                msg.show()
                return
            
        valid_dims = False
        resx = -1
        resy = -1
        config_debug(extent_type)
        # Show warning message extent limits extent_type
        if extent_type == "Current View":
            resx, resy = qgis_utils_get_current_mapview_dims()
            valid_dims = qgis_utils_valid_dims(resx, resy)
            config_debug("custom view",resx,resy)
        elif extent_type == "Layer extent":
            layer = parameters["layer"]
            resx, resy = qgis_utils_get_layer_dims(layer)
            valid_dims = qgis_utils_valid_dims(resx, resy)
            
        config_debug("dimensions",resx,resy)
            
        if not valid_dims:
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Current extent dimensions ({}x{}) are too big for processing.\nSmaller dimensions are recommended. Do you still want to process?".format(resx,resy))
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
            ret = msg.exec()          

            if ret == QMessageBox.No:
                return


        # Check what type of processing

        img = None

        temp_already_saved = False

        input_raster_path = None

        if extent_type == "Current View":
            # if capture canvas 
            img = self._capture_canvas(parameters["layer"], white_background = True)
            #img = self._capture_canvas(parameters["layer"], visible=True) # to save

            from .process.tree.interface.cachemanager import CacheManager
            cache_manager = CacheManager(project_path = parameters["output_path"])
            cache_dir = cache_manager.get_cache_folder()

            temp_raster = cache_manager.get_temp_raster_path()
            input_raster_path = temp_raster

            #img_bgr = img

            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

            extent = self.iface.mapCanvas().extent()
            epsg = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

            #from .process.tree.utils.utils_custom import np2tif_extent

            #np2tif_extent(img, extent, epsg, temp_raster)

            from .process.tree.utils.utils_custom import np2tif_extent_gdal
            np2tif_extent_gdal(img, extent, epsg, temp_raster)

            parameters['temp_raster'] = temp_raster

            parameters["is_temporal"] = True

            print(img.shape)
            
        elif extent_type == "Layer extent":
            
            layer = parameters["layer"]
            img = None

            extent = layer.extent()
            epgs = parameters["extent_crs"].authid()
            
            layer_path = layer.dataProvider().dataSourceUri()
            input_raster_path = layer_path
            
            #img = cv.imread(layer_path)

            #Correct extent
            # Create a QgsCoordinateTransform object
            crs_source = layer.crs()  # The source CRS of the layer
            crs_destination = QgsCoordinateReferenceSystem("EPSG:4326")  # The destination CRS
            transform = QgsCoordinateTransform(crs_source, crs_destination, QgsProject.instance())

            # Transform the extent to EPSG:4326
            extent_transformed = transform.transformBoundingBox(extent)

            extent = extent_transformed

        elif extent_type == "Custom extent":

            layer = parameters["layer"]
            img = None

            extent = parameters["extent"]
            epsg = parameters["extent_crs"].authid()

            
            layer_path = layer.dataProvider().dataSourceUri()
            temp_raster = os.path.join(parameters["output_path"],DEFAULT_TEMP_RASTER)
            input_raster_path = temp_raster

            raster_extract(layer_path, extent, epsg, temp_raster)

            img = cv.imread(temp_raster)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
            
            # Check dims for this case
            valid_dims = qgis_utils_valid_dims(resx, resy)            
            if not valid_dims:
                msg = QMessageBox(self.iface.mainWindow())
                msg.setWindowTitle("Tree Eyed")
                msg.setText("Current extent dimensions ({}x{}) are too big for processing.\nSmaller dimensions are recommended. Do you still want to process?".format(resx,resy))
                msg.setIcon(QMessageBox.Warning)
                msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
                ret = msg.exec()          

                if ret == QMessageBox.No:
                    self._remove_temp_raster(parameters["output_path"])
                    return

            temp_already_saved = True

        # **********************************************
        
        if is_task:
            
            # Read existing
            model_dir = self._get_models_dir()

            parameters["model_dir"] = model_dir

            if model_dir != "NODATA":
            
                # tree_predictor_task = TreePredictorTask("Tree predictor task", model_dir, parameters, img, extent, espg, temp_already_saved = temp_already_saved)
                # QgsApplication.taskManager().addTask(tree_predictor_task)
                # QgsMessageLog.logMessage("Inference process started", MESSAGE_CATEGORY, Qgis.Warning)
                # tree_predictor_task.task_finished.connect(self._process_task_finished)

                # fix necessary 
                parameters['task'] = "inference"
                parameters['input_raster_path'] = input_raster_path

                print()

                minx = extent.xMinimum()
                miny = extent.yMinimum()
                maxx = extent.xMaximum()
                maxy = extent.yMaximum()
                bounds = (minx, miny, maxx, maxy)

                parameters["extent_general"] = bounds
                parameters['epsg'] = parameters["extent_crs"].authid()

                parameters["output_files"] = []

                print(parameters)




                qgstask = WorkerTask("Tree inference task", parameters)
                qgstask.task_finished.connect(self._process_task_finished)
                QgsApplication.taskManager().addTask(qgstask)   
            
            else:
                print("NO modeldir")
            
            # progress.setValue(100)
        
            return
            
        else:        
            # img_bgr = self.predictor.predict(parameters, img, extent, espg)
            #img_bgr = self.predictor.predict_with_parameters(parameters, img, extent, epsg, temp_already_saved = temp_already_saved)

            #save capture
            #img2 = self._capture_canvas(parameters["layer"], visible=True)
            #self.predictor.save_capture(img, extent, espg)
            print("ELSE")

        # **********************************************

        #Visualize
        # window_name = "Inference"
        # h = img_bgr.shape[0]
        # w = img_bgr.shape[1]
        # #print(img_bgr.shape)
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_name, w, h)
        # cv.imshow(window_name, img_bgr)
        # cv.waitKey(1)


        # Add created files
        #if parameters['model'] == 'HighResCanopyHeight':

        print("output_files")
        print(self.predictor.output_files)

        self._add_processed_layers(self.predictor.output_files)

        temp_raster = os.path.join(parameters["output_path"],DEFAULT_TEMP_RASTER)
        if os.path.exists(temp_raster):
            os.remove(temp_raster)

        # progress.setValue(100)

        #progress.close()
        
    def _get_models_dir(self):
        """returns current model directory from QgsSettings for TreeEyed/modelDir key

        Returns:
            string: current model directory
        """
        
        # Read existing
        s = QgsSettings()
        plugin_name = "TreeEyed"
        model_dir = s.value(plugin_name + "/modelDir", "NODATA")
        
        return model_dir

    def _process_task_finished(self, results):
        """called when the processing task (inference) finishes

        Args:
            results (list): list of results filenames
        """
        
        QgsMessageLog.logMessage("Processing successful!",MESSAGE_CATEGORY, Qgis.Success)
        
        #img_bgr = results["img"]
        output_files = results["output_files"]
        output_path = results["output_path"]

        #print("_process_task_finished")
        #print(img_bgr)
        
        # #Visualize
        # window_name = "Inference"
        # h = img_bgr.shape[0]
        # w = img_bgr.shape[1]
        # #print(img_bgr.shape)
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_name, w, h)
        # cv.imshow(window_name, img_bgr)
        # cv.waitKey(1)

        #print("output_files")
        #print(output_files)

        self._add_processed_layers(output_files)

        # temp_raster = os.path.join(output_path,DEFAULT_TEMP_RASTER)
        # if os.path.exists(temp_raster):
        #     os.remove(temp_raster)
        self._remove_temp_raster(output_path)
            
    def _remove_temp_raster(self, output_path):
        """removes temporal raster created for processing

        Args:
            output_path (string): path of temp_raster
        """
        temp_raster = os.path.join(output_path,DEFAULT_TEMP_RASTER)
        if os.path.exists(temp_raster):
            os.remove(temp_raster)

    def _capture_canvas(self, selected_layer = None, visible = False, white_background = True):
        """captures canvas for processing

        Args:
            selected_layer (layer, option): If only selected layer. Defaults to None.
            visible (bool): If only visible layers. Defaults to False.
            white_background (bool, optional): Defaults to True.

        Returns:
            _type_: _description_
        """

        try:

            #print(self.iface.mapCanvas().size())
            width = self.iface.mapCanvas().size().width()
            height = self.iface.mapCanvas().size().height()

            img = QImage(QSize(width,height), QImage.Format_ARGB32_Premultiplied)
            #print(img)

            #set backgroundcolor
            if white_background:
                color = QColor(255,255,255,255)
            else:
                color = QColor(0,0,0,255)
            img.fill(color.rgba())
            

            #create painter
            p= QPainter()
            p.begin(img)
            p.setRenderHint(QPainter.Antialiasing)

            #the mapsettings
            ms= QgsMapSettings()
            ms.setBackgroundColor(color)

            #set layers to render
            #layer = QgsProject().instance().mapLayersByName('214')
            #ms.setLayers([layer[0]])

            config_debug(QgsProject.instance().mapLayers())
            config_debug(QgsProject.instance().mapLayers().values())

            layers = list(QgsProject.instance().mapLayers().values())
            
            config_debug("layers",layers)
            
            #print("layers added")

            #vlayer = self.iface.activeLayer()
            #vlayer = layers[1]
            #ms.setLayers([layers[1], layers[0]])

            #if not (selected_layer == None):
            #ms.setLayers(selected_layer)

            #layer = QgsProject.instance().mapLayersByName(selected_layer)

            layer_tree_root = QgsProject.instance().layerTreeRoot()

            layers = []
            #for temp_layer in list(QgsProject.instance().mapLayers().values()):
            for temp_layer in list(QgsProject().instance().layerTreeRoot().layerOrder()):
                config_debug(temp_layer.id())
                #print(selected_layer.id())

                if visible:

                    layer_tree_layer = layer_tree_root.findLayer(temp_layer.id())
                    layer_is_visible = layer_tree_layer.isVisible()
                    config_debug("VISIBLE", layer_is_visible)
                    if layer_is_visible:
                        layers.append(temp_layer)
                        #layers.insert(0,temp_layer)
                else:
                    if temp_layer.id() == selected_layer.id():
                        layers.append(temp_layer)

            #layer = QgsProject.instance().layerTreeRoot().findLayer(selected_layer.id())

            #self.iface.mapCanvas().setLayers(layers)

            config_debug("CURRENT LAYERS",ms.layers())
            #print(selected_layer)
            ms.setLayers(layers)
            config_debug("TO RENDER",ms.layers())

            #self.iface.mapCanvas().setLayers(layers)
            #ms.setDestinationCrs(layers[0].crs())
            config_debug("CRS", QgsProject.instance().crs())
            # It is necessary to set the CRS for multiple layers
            # Problems with custom CRS?
            # Add always a basemap first?
            ms.setDestinationCrs(QgsProject.instance().crs())


            #set Extent
            extent = self.iface.mapCanvas().extent()
            #rect = QgsRectangle(ms.fullExtent())
            #ms.setExtent(rect)
            ms.setExtent(extent)

            #size of output image
            ms.setOutputSize(img.size())

            #render map
            render = QgsMapRendererCustomPainterJob(ms, p)
            
            # render.prepare()
            # render.renderPrepared()

            #print("About to start renderer")
            render.start()
            #print("renderer started")
            render.waitForFinished()
            #print("renderer waited")
            p.end()
            #print("renderer finish")

            #save the image

            img_mat = self._QImageToCvMat(img)

            #print(img_mat)

            return img_mat
        
        except:
            print("Unkown error")
            return None
        
    def _QImageToCvMat(self, incomingImage):
        '''  Converts a QImage into an opencv MAT format  '''

        

        #incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        #Format_RGBA8888
        #incomingImage = incomingImage.copy().convertToFormat(QImage.Format_RGBA8888)

        width = incomingImage.width()
        height = incomingImage.height()
        #print("width", width)
        #print("height", height)
        

        #print("incomingImage", incomingImage)

        

        # ptr = incomingImage.constBits()
        # #ptr.setsize(height * width * 4)
        # #arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        # print(ptr)

        # #arr = np.array(ptr)#.reshape(height, width, 4)  #  Copies the data

        # #print(arr.shape)

        # ptr = image.constBits().asstring(width * height * 4)

        # arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        # print(arr)


        #ptr = incomingImage.constBits()
        #ptr.setsize(height * width * 4)
        #arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        #arr = np.frombuffer(arr)

        map_image = incomingImage

        height, width, bytesPerLine = map_image.height(), map_image.width(), map_image.bytesPerLine()
        img_bytes = map_image.bits().asstring(width * height * 4)  # Assuming RGB888 format
        #mat = cv.Mat(height, width, cv.CV_8UC4, img_bytes)
        arr = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 4))

        #print(arr)

        return arr
    
    def _prompt_download_models(self):
        """prompts warning to download AI models
        """
        
        # Check if download models
        msg = QMessageBox(self.iface.mainWindow())
        msg.setWindowTitle("Tree Eyed")
        msg.setText("Do you want to download the AI models? It may take a while.")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        ret = msg.exec()

        if ret == QMessageBox.Yes:
            
            QgsMessageLog.logMessage("Downloading models", MESSAGE_CATEGORY, Qgis.Warning)

            save_model_dir = str(QFileDialog.getExistingDirectory(self.iface.mainWindow(), "Select Directory"))
            
            if os.path.exists(save_model_dir):

                # Open log messages
                qgis_utils_show_log_messages_panel()

                # Run download
                model_downloader_task = ModelDownloaderTask("Tree Eyed downloading models", save_model_dir)
                QgsApplication.taskManager().addTask(model_downloader_task)
                QgsMessageLog.logMessage("Downloading models started", MESSAGE_CATEGORY, Qgis.Warning)

            else:
                QgsMessageLog.logMessage("Directory is not valid", MESSAGE_CATEGORY, Qgis.Warning)

        elif ret == QMessageBox.No:
            QgsMessageLog.logMessage("Downloading models canceled",MESSAGE_CATEGORY, Qgis.Warning)
        # ********************************
        
    def _check_valid_model(self, parameters):
        """checks if selected model is available

        Args:
            parameters (dict): contains the dict with the processing parameters

        Returns:
            Bool: True is selected model is available, otherwise False
        """
        
        model = parameters["model"]
        
        models_dict = {
            "Mask R-CNN":["MASKRCNNModel.onnx"] #MASKRCNN
            ,"HighResCanopyHeight": [
                                    "HRCH_model/HRCH_SSLhuge_satellite.onnx"
                                    #"compressed_SSLlarge.pth" #SSLlarge
                                    #,"compressed_SSLhuge_aerial.pth" #Huge Aerial
                                    #,"aerial_normalization_quantiles_predictor.ckpt" #normalization
                                    ] 
            ,"DeepForest": [
                           #"NEON.pt"
                            "DeepForestModel.onnx"
                            ]#Neon
            , "VHRTrees": ["VHRTrees_best.onnx"]
            , "Custom ONNX Model": []
        }
        
        required_files = models_dict[model]
        
        valid = True
        file_not_found = ""
        model_dir = self._get_models_dir()
        
        for file in required_files:            
            model_file = os.path.join(model_dir, file)
            if not os.path.exists(model_file):
                valid = False
                file_not_found = "\n{} not found in {}.".format(file, model_dir) 
                break
            
        if not valid:
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Cannot load the model." +  file_not_found+ "\nPlease make sure the model files are available in the models directory (Settings)")
            msg.setIcon(QMessageBox.Critical)
            msg.show()
            
        return valid

    def _check_current_tasks(self):
        """checks if there are tasks already running

        Returns:
            Bool: returns True if cannot run new task, otherwise returns False
        """
        
        tasks = QgsApplication.taskManager().activeTasks()
        valid = True
        
        for task in tasks:
            description = task.description()
            config_debug("running task", description)
            if description == "Tree predictor task":
                valid = False
                break            
            
        if not valid:
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("There is a processing task already running. Please wait until it finishes to start a new task.")
            msg.setIcon(QMessageBox.Information)
            msg.show()
            
        return valid
    
    def _check_result_types(self, parameters):
        """checks that there are valid result types selected 

        Args:
            parameters (dict): contains the dict with the processing parameters

        Returns:
            Bool: returns True if valid, otherwise False.
        """
        
        #valid = self.iface.mainWindow()._result_types_are_selected()
        
        valid = True
        total_outputs = len(parameters["raster_outputs"]) + len(parameters["vector_outputs"])
        
        if  total_outputs <= 0:
            valid = False
        
        if not valid:
            msg = QMessageBox(self.iface.mainWindow())
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Please select at least one result type.")
            msg.setIcon(QMessageBox.Information)
            msg.show()
            
        return valid
            
           
    
class ModelDownloaderTask(QgsTask):
    """This class is used as a helper to perform the downloading of the models
    """

    def __init__(self, description, dir_models):
        """Constructor

        Args:
            description (string): description of the task
            dir_models (string): directory of the models
        """
        super().__init__(description, QgsTask.CanCancel)

        self.dir_models = dir_models

        # # Hardcoded model urls
        # self.urls = [
        #     "https://drive.google.com/file/d/1TQtmmj8M3Slrs_zTaVyXJOKGEzZqq3VG/view?usp=drive_link" #MASKRCNN
        #     ,"https://drive.google.com/file/d/191KeFSxNc-liH9eEn9pUmGgyF4q5VL1d/view?usp=drive_link" #SSLlarge
        #     ,"https://drive.google.com/file/d/1ixyi9AB6S4Qawl4pPJaI3-2iijJGxoKA/view?usp=drive_link" #Huge Aerial
        #     ,"https://drive.google.com/file/d/1yBM3pb4tKg5XSfPf77VGkTuKK39mYPO7/view?usp=drive_link" #normalization
        #     ,"https://drive.google.com/file/d/1MzBhE5N6KVEKWc-_ryLn7fi7P6kiyn6e/view?usp=drive_link"#Neon  
        # ]

        # self.urls_names = [
        #     "MASKRCNNModel.pth" #MASKRCNN
        #     ,"compressed_SSLlarge.pth" #SSLlarge
        #     ,"compressed_SSLhuge_aerial.pth" #Huge Aerial
        #     ,"aerial_normalization_quantiles_predictor.ckpt" #normalization
        #     ,"NEON.pt"#Neon  
        # ]

        # Hardcoded model urls for ONNX models
        self.urls = [
            #"https://drive.google.com/drive/folders/1diJ5so9FjwFi45phV-NOnuuL_6Xdx80x?usp=drive_link" #HRCH ONNX
             "https://drive.google.com/file/d/1JhpsEjglsmmlN0kNCRqp1xAXohUCc4Xm/view?usp=drive_link" #MASKRCNN ONNX
            , "https://drive.google.com/file/d/17cpB49Sy1GNmkhFgSsuyRT_4Yx44nI0W/view?usp=drive_link" #DeepForest ONNX
            , "https://drive.google.com/file/d/1cjrwD6SekWgXEhOGF9bRQuQ721_BKOEJ/view?usp=drive_link" #VHRTrees ONNX
            ]

        self.urls_names = [
            #"HRCH_model" #HRCH ONNX folder
            "MASKRCNNModel.onnx" #MASKRCNN ONNX
            ,"DeepForestModel.onnx" #DeepForest ONNX
            , "VHRTrees_best.onnx" #VHRTrees ONNX
        ]

                # Hardcoded model urls for ONNX models
        self.urls_folders = [
            "https://drive.google.com/drive/folders/1diJ5so9FjwFi45phV-NOnuuL_6Xdx80x?usp=drive_link" #HRCH ONNX
           ]

        self.urls_folders_names = [
            "HRCH_model" #HRCH ONNX folder
        ]

    def run(self):
        """executes the downloading task

        Returns:
            Bool: returns True if task was successful
        """

        #self.setProgress(10)

        QgsMessageLog.logMessage('Started task "{}"'.format(
                                     self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)

        for index,url in enumerate(self.urls):

            step_progress = (index)*1.0/(len(self.urls)+len(self.urls_folders))*100
            self.setProgress(step_progress)

            model_filepath = os.path.join(self.dir_models,self.urls_names[index])
            QgsMessageLog.logMessage("Downloading " + url+ " " + model_filepath,MESSAGE_CATEGORY, Qgis.Info)
            import gdown
            if not os.path.exists(model_filepath):
                gdown.download(url, output=model_filepath, fuzzy=True)

            if self.isCanceled():
                return False
            
        for index,url in enumerate(self.urls_folders):

            step_progress = (index+len(self.urls))*1.0/(len(self.urls)+len(self.urls_folders))*100
            self.setProgress(step_progress)

            model_filepath = os.path.join(self.dir_models,self.urls_folders_names[index])
            QgsMessageLog.logMessage("Downloading " + url+ " " + model_filepath,MESSAGE_CATEGORY, Qgis.Info)
            import gdown
            if not os.path.exists(model_filepath):
                gdown.download_folder(url, output=model_filepath, quiet=False)

            if self.isCanceled():
                return False

        self.setProgress(100)

        return True

    def finished(self, result):
        """executed on finishing the task, shows messages

        Args:
            result (Bool): successful task
        """

        if result:
            QgsMessageLog.logMessage("Model download successful!",MESSAGE_CATEGORY, Qgis.Success)
            print("reloading")
            qgis.utils.reloadPlugin("tree_eyed")
        else:
            QgsMessageLog.logMessage("Installation was not successful!",MESSAGE_CATEGORY, Qgis.Critical)

    def cancel(self):
        """cancel task
        """
        QgsMessageLog.logMessage('Package installation was canceled',MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()
        
        
        

