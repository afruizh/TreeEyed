# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
# Initialize Qt resources from file resources.py
from .resources import *

# Import the code for the DockWidget
from .tree_eyed_dockwidget import TreeEyedDockWidget
import os.path

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
)

from qgis.core import Qgis

#from qgis.analysis import QgsZonalStatistics

from .process.gui_utils.config import *
from .process.gui_utils.qgis_utils import *

from .process.tree.metrics.compare import COCOMetrics

from time import sleep
import random
MESSAGE_CATEGORY = 'Tree Eyed Plugin'


#from .process.gui_utils.installer import InstallerManager

import cv2 as cv
import numpy as np
import glob
from .process.tree.tree_predictor_task import TreePredictorTask
import os

from .process.tree.utils.utils_custom import *
from .process.tree.qgis2coco.qgis2coco import *

from pathlib import Path

import gdown
import sys

os.environ["TQDM_DISABLE"] = "1" 

class NullWriter:
    def write(self, data):
        pass

# Override stdout and stderr with NullWriter in GUI --noconsole mode
# This allow to avoid a bug where tqdm try to write on NoneType
if sys.stdout is None:
    sys.stdout = NullWriter()

if sys.stderr is None:
    sys.stderr = NullWriter()

DEFAULT_RASTER_COLORMAP = "colormap.txt"
DEFAULT_VECTOR_BB = "detection_style.qml"
DEFAULT_TEMP_RASTER = "_tree_eyed_temp_raster.tif"

class TreeEyed:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # update default paths
        global DEFAULT_RASTER_COLORMAP
        DEFAULT_RASTER_COLORMAP = os.path.join(self.plugin_dir, "colormap.txt")
        global DEFAULT_VECTOR_BB
        DEFAULT_VECTOR_BB = os.path.join(self.plugin_dir, "detection_style.qml")

        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'TreeEyed_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&TreeEyed')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'TreeEyed')
        self.toolbar.setObjectName(u'TreeEyed')

        #print "** INITIALIZING TreeEyed"

        self.pluginIsActive = False
        self.dockwidget = None
        
        # Read existing settings
        model_dir = self._get_models_dir()
        
        if model_dir == "NODATA":            
            self._prompt_download_models()


    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('TreeEyed', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action


    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/tree_eyed/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Tree Eyed'),
            callback=self.run,
            parent=self.iface.mainWindow())

    #--------------------------------------------------------------------------

    def onClosePlugin(self):
        """Cleanup necessary items here when plugin dockwidget is closed"""

        #print "** CLOSING TreeEyed"

        # disconnects
        self.dockwidget.closingPlugin.disconnect(self.onClosePlugin)

        # remove this statement if dockwidget is to remain
        # for reuse if plugin is reopened
        # Commented next statement since it causes QGIS crashe
        # when closing the docked window:
        # self.dockwidget = None

        self.pluginIsActive = False


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""

        #print "** UNLOAD TreeEyed"

        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&TreeEyed'),
                action)
            self.iface.removeToolBarIcon(action)
        # remove the toolbar
        del self.toolbar

    #--------------------------------------------------------------------------

    def run(self):
        """Run method that loads and starts the plugin"""

        if not self.pluginIsActive:
            self.pluginIsActive = True

            #print "** STARTING TreeEyed"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = TreeEyedDockWidget()

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)


            # Additional Connections
            self.dockwidget.process_signal.connect(self._process)
            #self.dockwidget.process_signal.connect(self._process_task)
            self.iface.mapCanvas().scaleChanged.connect(self.dockwidget._handle_mapScaleChanged)
            
            self.dockwidget.download_models_signal.connect(self._prompt_download_models)

            # show the dockwidget
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
            self.dockwidget.show()

    #--------------------------------------------------------------------------
    # Additional functions

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
        
        area_value = parameters["filter_area_area"]
        
        
        selected_layer = parameters["filter_area_layer"]
        layer_path = selected_layer.dataProvider().dataSourceUri()
        
        gdf = gpd.read_file(layer_path)       
        gdf = gdf[gdf["area_m2"] <= area_value]
        
        output_dir = parameters["output_path"]
        output_prefix = parameters["prefix"]
        output_filename = os.path.join(output_dir, output_prefix + "_vector.shp")
        
        
        gdf.to_file(output_filename, index=False)
        
        layers = self._add_processed_layers([output_filename])
        
        return
    
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

        img = cv.imread(layer_path, cv.IMREAD_GRAYSCALE)
        #config_debug(img.dtype)
        #config_debug(img.shape)
        config_debug("max",np.max(img))
        
        percentage = parameters["raster2vector_threshold"]/100.0
        
        max_value = np.max(img)
        value = percentage*max_value
        ret, img = cv.threshold(img, value, 255, 0)
        
        config_debug("max",np.max(img))
        
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
        
        

        # extent = self.iface.mapCanvas().extent()
        epsg = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

        # img = self._capture_canvas(parameters["layer"], visible=True)
        model_dir = self._get_models_dir()
        
        tree_predictor_task = TreePredictorTask("Tree predictor task", model_dir, parameters, img, extent, epsg, temp_already_saved = False)

        tree_predictor_task.output_dir = parameters["output_path"]
        tree_predictor_task.output_prefix = parameters["prefix"]
        tree_predictor_task.output_filename = os.path.join(tree_predictor_task.output_dir, tree_predictor_task.output_prefix + "_vector.shp")
               
        tree_predictor_task.output_files = []
        #tree_predictor_task.output_files.append(tree_predictor_task.output_filename)
        
        

        tree_predictor_task.save_shapefile_polygon_binary_raster(img
                                                       , extent
                                                       , img.shape[1]
                                                       , img.shape[0]
                                                    , epsg
                                                       )
        
        layers = self._add_processed_layers(tree_predictor_task.output_files)


        # # Calculate zonal statistics
        # # Create zonal statistics object
        # zonal_stats = QgsZonalStatistics(layers[0], selected_layer
        #                                 , attributePrefix="height"
        #                                 , rasterBand=1, stats=QgsZonalStatistics.Statistics(QgsZonalStatistics.Max)
        #                                 )

        # # Configure statistics
        # #zonal_stats.setStatistics(QgsZonalStatistics.Mean)

        # # Calculate zonal statistics
        # zonal_stats.calculateStatistics(None)
        # #zonal_stats.calculateStatistics(QgsZonalStatistics.SecondPass)

        # # for field in layer.fields():
        # # if field.name() == 'old_fieldname':

        # #     with edit(layer):
        # #         idx = layer.fields().indexFromName(field.name())
        # #         layer.renameAttribute(idx, 'new_fieldname')

    def _process_validate(self, parameters):
        """calculates validation metrics betwee 2 COCO datasets in .json format

        Args:
            parameters (dict): contains the dict with the processing parameters
        """
        
        validate_ground_truth = parameters["validate_ground_truth"]
        validate_prediction = parameters["validate_prediction"]
        
        if os.path.exists(validate_ground_truth) and os.path.exists(validate_prediction):
        
            coco_metrics = COCOMetrics()
            coco_metrics.load_target(validate_ground_truth, result_type='coco')
            coco_metrics.load_pred(validate_prediction, result_type='coco')
            coco_metrics.compute()
            
            msg = QMessageBox(self.dockwidget)
            msg.setWindowTitle("Tree Eyed")
            msg.setText(coco_metrics.final_message)
            msg.setIcon(QMessageBox.Information)
            msg.show()
            
        else:
            msg = QMessageBox(self.dockwidget)
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
                
                t_image = cv.imread(file)
                max_value = np.max(t_image)
                config_debug("max_value", max_value)
                

                #https://docs.qgis.org/3.34/en/docs/pyqgis_developer_cookbook/raster.html
                fcn = QgsColorRampShader()
                fcn.setColorRampType(QgsColorRampShader.Interpolated)
                color_ramp_load = QgsRasterRendererUtils.parseColorMapFile(DEFAULT_RASTER_COLORMAP)
                #lst = [ QgsColorRampShader.ColorRampItem(0, QColor(0,255,0)),
                #    QgsColorRampShader.ColorRampItem(255, QColor(255,255,0))]
                lst0 = color_ramp_load[1]
                
                color0 = lst0[0].color
                color1 = lst0[1].color
                config_debug("colors")
                config_debug(color0.red(), color0.green(), color0.blue(), color0.alpha())
                config_debug(color1.red(), color1.green(), color1.blue(), color1.alpha())
                
                color = QColor(19,222,222,255)
                
                lst = []
                lst.append(lst0[0])
                lst.append(QgsColorRampShader.ColorRampItem(max_value,color,lst0[1].label))

                fcn.setColorRampItemList(lst)
                shader = QgsRasterShader()
                shader.setRasterShaderFunction(fcn)                
                
                renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
                layer.setRenderer(renderer)


                if not layer.isValid():
                    print("Layer failed to load!", )
                else:
                    QgsProject.instance().addMapLayer(layer)
                    layers.append(layer)
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
            msg = QMessageBox(self.dockwidget)
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Please select an output directory")
            msg.setIcon(QMessageBox.Information)
            msg.show()
            return
        
        # Show warning message if outputs already exists
        if (self._check_already_exist(parameters)):
            msg = QMessageBox(self.dockwidget)
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

            img = self._capture_canvas(parameters["layer"], visible=True)

            img_bgr = img

            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

            extent = self.iface.mapCanvas().extent()
            epgs = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

            capture_raster = os.path.join(parameters["output_path"],"capture_raster.tif")

            # Save current raster
            if extent is not None:
                #np2tif_extent(np_image, extent, epsg, DEFAULT_TEMP_OUTPUT_RASTER)
                np2tif_extent(img, extent, epgs, capture_raster)


            return
        
        if parameters['task'] == "export_dataset":

            image_path = parameters["input_image"].dataProvider().dataSourceUri()
            annotations_path = parameters["annotations"].dataProvider().dataSourceUri()
            num_tiles = parameters["num_tiles"]
            
            dir_name = parameters["prefix"] + "_coco_dataset"
            
            path_output = os.path.join(parameters["output_path"], dir_name)

            #Check if already exist
            if os.path.exists(path_output):
                msg = QMessageBox(self.dockwidget)
                msg.setWindowTitle("Tree Eyed")
                msg.setText("Output directory {} already exists. Please select a different output name.".format(dir_name))
                msg.setIcon(QMessageBox.Information)
                msg.show()
                return

            exporter = QGIS2COCO(image_path, annotations_path)
            exporter.convert(path_output, num_tiles, 1.0)

            return
        
        # Show warning message custom extent for WMS layer
        if extent_type == "Custom extent":

            layer = parameters["layer"]

            if layer.providerType() == 'wms':
                msg = QMessageBox(self.dockwidget)
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
            msg = QMessageBox(self.dockwidget)
            msg.setWindowTitle("Tree Eyed")
            msg.setText("Current extent dimensions ({}x{}) are too big for processing.\nSmaller dimensions are recommended. Do you want still want to process?".format(resx,resy))
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
            ret = msg.exec()          

            if ret == QMessageBox.No:
                return


        # Check what type of processing

        img = None

        temp_already_saved = False

        if extent_type == "Current View":
            # if capture canvas 
            img = self._capture_canvas(parameters["layer"])
            #img = self._capture_canvas(parameters["layer"], visible=True) # to save

            img_bgr = img

            img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)

            extent = self.iface.mapCanvas().extent()
            espg = self.iface.mapCanvas().mapSettings().destinationCrs().authid()

            print(img.shape)
            
        elif extent_type == "Layer extent":
            
            layer = parameters["layer"]
            img = None

            extent = parameters["extent"]
            espg = parameters["extent_crs"].authid()
            
            layer_path = layer.dataProvider().dataSourceUri()
            
            img = cv.imread(layer_path)
            

            return
        elif extent_type == "Custom extent":

            layer = parameters["layer"]
            img = None

            extent = parameters["extent"]
            espg = parameters["extent_crs"].authid()

            
            layer_path = layer.dataProvider().dataSourceUri()
            temp_raster = os.path.join(parameters["output_path"],DEFAULT_TEMP_RASTER)

            raster_extract(layer_path, extent, espg, temp_raster)

            img = cv.imread(temp_raster)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
            
            # Check dims for this case
            valid_dims = qgis_utils_valid_dims(resx, resy)            
            if not valid_dims:
                msg = QMessageBox(self.dockwidget)
                msg.setWindowTitle("Tree Eyed")
                msg.setText("Current extent dimensions ({}x{}) are too big for processing.\nSmaller dimensions are recommended. Do you want still want to process?".format(resx,resy))
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

            if model_dir != "NODATA":
            
                tree_predictor_task = TreePredictorTask("Tree predictor task", model_dir, parameters, img, extent, espg, temp_already_saved = temp_already_saved)
                QgsApplication.taskManager().addTask(tree_predictor_task)
                QgsMessageLog.logMessage("Inference process started", MESSAGE_CATEGORY, Qgis.Warning)
                tree_predictor_task.task_finished.connect(self._process_task_finished)
            
            else:
                print("NO modeldir")
            
            # progress.setValue(100)
        
            return
            
        else:        
            # img_bgr = self.predictor.predict(parameters, img, extent, espg)
            img_bgr = self.predictor.predict_with_parameters(parameters, img, extent, espg, temp_already_saved = temp_already_saved)

            #save capture
            #img2 = self._capture_canvas(parameters["layer"], visible=True)
            #self.predictor.save_capture(img, extent, espg)

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
        
        img_bgr = results["img"]
        output_files = results["output_files"]
        output_path = results["output_path"]

        print("_process_task_finished")
        print(img_bgr)
        
        # #Visualize
        # window_name = "Inference"
        # h = img_bgr.shape[0]
        # w = img_bgr.shape[1]
        # #print(img_bgr.shape)
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_name, w, h)
        # cv.imshow(window_name, img_bgr)
        # cv.waitKey(1)

        print("output_files")
        print(output_files)

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
            "Mask R-CNN":["MASKRCNNModel.pth"] #MASKRCNN
            ,"HighResCanopyHeight": ["compressed_SSLlarge.pth" #SSLlarge
            ,"compressed_SSLhuge_aerial.pth" #Huge Aerial
            ,"aerial_normalization_quantiles_predictor.ckpt"] #normalization
            ,"DeepForest": ["NEON.pt"]#Neon  
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
            msg = QMessageBox(self.dockwidget)
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
            msg = QMessageBox(self.dockwidget)
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
        
        #valid = self.dockwidget._result_types_are_selected()
        
        valid = True
        total_outputs = len(parameters["raster_outputs"]) + len(parameters["vector_outputs"])
        
        if  total_outputs <= 0:
            valid = False
        
        if not valid:
            msg = QMessageBox(self.dockwidget)
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

        # Hardcoded model urls
        self.urls = [
            "https://drive.google.com/file/d/1TQtmmj8M3Slrs_zTaVyXJOKGEzZqq3VG/view?usp=drive_link" #MASKRCNN
            ,"https://drive.google.com/file/d/191KeFSxNc-liH9eEn9pUmGgyF4q5VL1d/view?usp=drive_link" #SSLlarge
            ,"https://drive.google.com/file/d/1ixyi9AB6S4Qawl4pPJaI3-2iijJGxoKA/view?usp=drive_link" #Huge Aerial
            ,"https://drive.google.com/file/d/1yBM3pb4tKg5XSfPf77VGkTuKK39mYPO7/view?usp=drive_link" #normalization
            ,"https://drive.google.com/file/d/1MzBhE5N6KVEKWc-_ryLn7fi7P6kiyn6e/view?usp=drive_link"#Neon  
        ]

        self.urls_names = [
            "MASKRCNNModel.pth" #MASKRCNN
            ,"compressed_SSLlarge.pth" #SSLlarge
            ,"compressed_SSLhuge_aerial.pth" #Huge Aerial
            ,"aerial_normalization_quantiles_predictor.ckpt" #normalization
            ,"NEON.pt"#Neon  
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

            step_progress = (index)*1.0/len(self.urls)*100
            self.setProgress(step_progress)

            model_filepath = os.path.join(self.dir_models,self.urls_names[index])
            QgsMessageLog.logMessage("Downloading " + url+ " " + model_filepath,MESSAGE_CATEGORY, Qgis.Info)
            if not os.path.exists(model_filepath):
                gdown.download(url, output=model_filepath, fuzzy=True)

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
        
    
        
        

