# -*- coding: utf-8 -*-

import os
import glob

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal, Qt, QUrl

# Additional imports
#from qgis.core import QgsRasterLayer
from qgis.core import QgsMapLayerProxyModel

from qgis.core import (
  QgsSettings
  , QgsCoordinateReferenceSystem
  , QgsRectangle
)

from qgis.utils import iface
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtGui import QDesktopServices

from .process.gui_utils.config import *

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'tree_eyed_dockwidget_base.ui'))

FORM_CLASS_SETTINGS_DIALOG, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'tree_eyed_settings_dialog.ui'))

import re

def increment_string_number(string, path, count = 0):
    """
        Increment the number in a string.

        If the string does not contain a number, add it with 3 leading 0, starting with 0001.

        Args:
        string: The string to increment.

        Returns:
        The incremented string.
    """

    # Find the last number in the string.
    match = re.search(r"\d+$", string)

    res = ""

    # If the string does not contain a number, add it with 3 leading 0, starting with 0001.
    if match is None:
        res =  string + "0001"    
    else: 

        # Increment the number.
        number = int(match.group()) + 1

        # Add leading 0s to the number.
        number_str = str(number).zfill(4)

        # Replace the old number with the new number.
        res =  string[:-len(match.group())] + number_str

    print(res)

    if count >= 1000:
        return res
    
    #Check already exists
    pattern = os.path.join(path, res + "_*")
    files = glob.glob(pattern)

    print(files)

    if len(files) > 0:
        return increment_string_number(res, path, count+1)
    
    return res

class TreeEyedDockWidget(QtWidgets.QDockWidget, FORM_CLASS):

    closingPlugin = pyqtSignal()

    # Additional signals
    process_signal = pyqtSignal(dict)
    download_models_signal = pyqtSignal()

    def __init__(self, parent=None):
        """Constructor."""
        super(TreeEyedDockWidget, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://doc.qt.io/qt-5/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        
        # GUI elements
        #pixmap = QPixmap(":/plugins/tree_eyed/logo.png").scaledToWidth(200, Qt.SmoothTransformation)  # Scale to desired width
        pixmap = QPixmap(":/plugins/tree_eyed/res/Banner TreeEyed.png").scaledToWidth(500, Qt.SmoothTransformation)  # Scale to desired width
        self.label_5.setPixmap(pixmap)
        self.label_5.setAlignment(Qt.AlignCenter)

        # Hide old titles
        self.label_4.setVisible(False)
        
        # pixmap = QPixmap(":/plugins/tree_eyed/forest.png").scaledToWidth(40, Qt.SmoothTransformation)  # Scale to desired width
        # self.label_5.setPixmap(pixmap)
        # self.label_5.setAlignment(Qt.AlignCenter)

        # Set icon to button pushButton_simple_inference
        icon = QtGui.QIcon(":/plugins/tree_eyed/res/icon_black.png")
        self.pushButton_simple_inference.setIcon(icon)
        # Assign slot to button
        self.pushButton_simple_inference.clicked.connect(self.run_simple_inference)

        # Configure combobox filters
        self.mMapLayerComboBox_inputLayer.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBox_input_image.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBox_raster2vector.setFilters(QgsMapLayerProxyModel.RasterLayer)
        
        self.mMapLayerComboBox_annotations.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.mMapLayerComboBox_filter_area.setFilters(QgsMapLayerProxyModel.VectorLayer)
        
        self.pushButton_test_process.setVisible(False)
        self.mMapLayerComboBox_validation_ground_truth.setVisible(False)
        self.mMapLayerComboBox_validation_prediction.setVisible(False)

        # Hide advanced input options not used
        #self.mGroupBox_advanced.setVisible(False)

        self.spinBox_tile_size.setEnabled(False)

        # enable or disable spinBox_tile_size
        self.checkBox_force_tiling.toggled.connect(self.spinBox_tile_size.setEnabled)
        

        self._initial_configuration()
        self._create_connections()
        self._update_model_options("HighResCanopyHeight")

        self.mExtentGroupBox.setOriginalExtent(QgsRectangle(0,0,0,0) , QgsCoordinateReferenceSystem('EPSG:4326'))
        self.mExtentGroupBox.setOutputCrs(QgsCoordinateReferenceSystem('EPSG:4326'))

        self.m_settings_dialog = SettingsDialog(self)
        self.m_settings_dialog.pushButton_download_models.clicked.connect(self.download_models_signal)

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()

    #Additional functions
    def _create_connections(self):
        """Creates the required connections for the dock widget
        """

        self.pushButton_process.clicked.connect(self._process)
        self.comboBox_model.currentTextChanged.connect(self._update_model_options)
        self.comboBox_extent.currentTextChanged.connect(self._update_extent_options)
        
        self.pushButton_settings.clicked.connect(self._open_settings)
        self.pushButton_raster2vector.clicked.connect(self._process_raster2vector)
        self.pushButton_test_process.clicked.connect(self._process_capture)
        self.pushButton_filter_area.clicked.connect(self._process_filter_area)

        self.pushButton_export_dataset.clicked.connect(self._process_export_dataset)
        self.pushButton_validate.clicked.connect(self._process_validate)

        self.mMapLayerComboBox_inputLayer.layerChanged.connect(self._handle_layerChanged)

        self.mQgsFileWidget.fileChanged.connect(self._handle_outputDirChanged)
        self.m_pushButton_increment.clicked.connect(self._handle_buttonIncrement)
        self.m_pushButton_openfolder.clicked.connect(self._handle_buttonOpenFolder)
        self.lineEdit_prefix.textChanged.connect(self._handle_outputNameChanged)

        self.m_pushButton_clear_cache.clicked.connect(self._handle_clean_cache_folder)

        # Map canvas collection
        print(self.parent)

    def _initial_configuration(self):
        self.mGroupBox_HRCH.setVisible(False)
        self.mExtentGroupBox.setVisible(False)
        self.mGroupBox_advanced.setCollapsed(True)

        # Load default output directory
        s = QgsSettings()
        plugin_name = "TreeEyed"
        output_dir = s.value(plugin_name + "/outputDir", "")
        if os.path.isdir(output_dir):
            self.mQgsFileWidget.setFilePath(output_dir)

        # Load default output name
        output_name = s.value(plugin_name + "/outputName", "")
        if output_name:
            self.lineEdit_prefix.setText(output_name)
        else:
            self.lineEdit_prefix.setText("results")

    def _open_settings(self):
        self.m_settings_dialog.open()

    def _handle_outputDirChanged(self, output_dir):

        if os.path.isdir(output_dir):
            s = QgsSettings()
            plugin_name = "TreeEyed"
            output_dir = s.setValue(plugin_name + "/outputDir", output_dir)

    def _handle_outputNameChanged(self, output_name):

        s = QgsSettings()
        plugin_name = "TreeEyed"
        output_dir = s.setValue(plugin_name + "/outputName", output_name)

    def _handle_buttonIncrement(self):

        name = self.lineEdit_prefix.text()

        name = increment_string_number(name, self.mQgsFileWidget.filePath())

        self.lineEdit_prefix.setText(name)

    def _handle_buttonOpenFolder(self):

        filepath = self.mQgsFileWidget.filePath()

        if os.path.isdir(filepath):
            QDesktopServices.openUrl(QUrl.fromLocalFile(filepath))

    def _handle_layerChanged(self, layer):

        config_debug("layer", layer)
        
        if layer is None:
            config_debug("layer is none")
            return

        extent = layer.extent()
        data_provider = layer.dataProvider()

        config_debug(data_provider.xSize())

        if (data_provider.xSize() > 0 and data_provider.ySize() > 0):

            xRes = extent.width() / data_provider.xSize()
            yRes = extent.height() / data_provider.ySize()

            self.label_spatial_resolution.setText("{0:.3f}".format(xRes) + "," + "{0:.3f}".format(yRes) + " w: " + str(data_provider.xSize()) + " h: " + str(data_provider.ySize()))
        else:
            self.label_spatial_resolution.setText("--")

    def _handle_mapScaleChanged(self, scale):

        self.label_spatial_resolution_map.setText(str(scale))
        #print(iface)

        extent = iface.mapCanvas().extent()
        width = iface.mapCanvas().size().width()
        height = iface.mapCanvas().size().height()

        if (width > 0 and height > 0):

            xRes = extent.width() / width
            yRes = extent.height() / height

            self.label_spatial_resolution_map.setText("{0:.3f}".format(xRes) + "," + "{0:.3f}".format(yRes) + " w: " + str(width) + " h: " + str(height))
        else:
            self.label_spatial_resolution_map.setText("--")

    def _process_raster2vector(self):
        self._process(task="raster2vector")
        
    def _process_filter_area(self):
        self._process(task="filter_area")

    def _process_capture(self):
        self._process(task="capture")

    def _process_export_dataset(self):
        self._process(task="export_dataset")
    
    def _process_validate(self):
        self._process(task="validate")

    def _process(self, task="inference"):

        # print("_process2")

        # print(self.comboBox_model.currentIndex())
        # print(self.comboBox_model.currentText())
        # print(self.lineEdit_prefix.text())
        # print(self.mQgsFileWidget.filePath())
        # print(self.mQgsFileWidget.filePath())
        # print(self.mMapLayerComboBox_inputLayer.currentLayer())

        raster_outputs = []
        vector_outputs = []

        checkbox_raster_dict = {"binary":self.checkBox_raster_binary
                             , "grayscale": self.checkBox_raster_grayscale}
        for key, checkbox in checkbox_raster_dict.items():
            if checkbox.isVisible() and checkbox.isChecked():
                raster_outputs.append(key)

        checkbox_vector_dict = {"polygons":self.checkBox_vector_polygons
                                , "bounding_boxes": self.checkBox_vector_bounding_boxes
                                , "centroids": self.checkBox_vector_centroids}
        for key, checkbox in checkbox_vector_dict.items():
            if checkbox.isVisible() and checkbox.isChecked():
                vector_outputs.append(key)

        parameters = {
            "model": self.comboBox_model.currentText()
            ,"output_path": self.mQgsFileWidget.filePath()
            ,"prefix":self.lineEdit_prefix.text()
            ,"hrch_type": self.comboBox_hrch_type.currentText()
            ,"hrch_threshold": self.mQgsDoubleSpinBox_hrch_threshold.value()
            ,"layer": self.mMapLayerComboBox_inputLayer.currentLayer()
            ,"raster2vector_layer": self.mMapLayerComboBox_raster2vector.currentLayer()
            ,"raster2vector_threshold": self.spinBox_raster2vector_threshold.value()
            ,"filter_area_layer":self.mMapLayerComboBox_filter_area.currentLayer()
            ,"filter_area_area": self.doubleSpinBox_filter_area.value()
            ,"task": task
            ,"raster_outputs": raster_outputs
            ,"vector_outputs": vector_outputs
            , "extent_type": self.comboBox_extent.currentText()
            , "extent": self.mExtentGroupBox.outputExtent()
            #, "extent_crs": self.mExtentGroupBox.currentCrs()
            , "extent_crs": self.mExtentGroupBox.originalCrs()
            , "input_image": self.mMapLayerComboBox_input_image.currentLayer()
            , "annotations": self.mMapLayerComboBox_annotations.currentLayer()
            , "num_tiles": self.spinBox_tiles.value()
            , "validate_ground_truth": self.mQgsFileWidget_validation_ground_truth.filePath()
            , "validate_prediction": self.mQgsFileWidget_validation_prediction.filePath()
            , "output_format": self.comboBox_output_format.currentText()
            , "overlap": self.spinBox_overlap.value()
            #, "tile_size": self.spinBox_tile_size.value()
            , "force_tiling": self.checkBox_force_tiling.isChecked()
            , "custom_model_filepath": self.mQgsFileWidget_model_filepath.filePath()
        }

        # Add tile size depending on checkBox_force_tiling if it's checked
        if self.checkBox_force_tiling.isChecked():
            parameters["tile_size"] = self.spinBox_tile_size.value()

        self.process_signal.emit(parameters)

    def _update_model_options(self, text):

        print(text)

        self.mGroupBox_HRCH.setVisible(False)
        self.mGroupBox_Custom.setVisible(False)
        self.mGroupBox_result_types.setVisible(True)


        # Enable all checkboxes
        self.checkBox_raster_binary.setEnabled(True)
        self.checkBox_raster_grayscale.setEnabled(True)
        self.checkBox_vector_polygons.setEnabled(True)
        self.checkBox_vector_bounding_boxes.setEnabled(True)
        self.checkBox_vector_centroids.setEnabled(True)

        if text == "HighResCanopyHeight":
            #self.mGroupBox_HRCH.setVisible(True)

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(True)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

            self.checkBox_raster_binary.setChecked(False)
            self.checkBox_raster_grayscale.setChecked(True)
            self.checkBox_vector_polygons.setChecked(False)
            self.checkBox_vector_bounding_boxes.setChecked(False)
            self.checkBox_vector_centroids.setChecked(False)

            self.checkBox_raster_grayscale.setEnabled(False)

        elif text == "Mask R-CNN":

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

            self.checkBox_raster_binary.setChecked(True)
            self.checkBox_raster_grayscale.setChecked(False)
            self.checkBox_vector_polygons.setChecked(False)
            self.checkBox_vector_bounding_boxes.setChecked(False)
            self.checkBox_vector_centroids.setChecked(False)

            self.checkBox_raster_binary.setEnabled(False)

        elif text == "DeepForest":

            self.checkBox_raster_binary.setVisible(False)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(False)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

            self.checkBox_raster_binary.setChecked(False)
            self.checkBox_raster_grayscale.setChecked(False)
            self.checkBox_vector_polygons.setChecked(False)
            self.checkBox_vector_bounding_boxes.setChecked(True)
            self.checkBox_vector_centroids.setChecked(False)

            self.checkBox_vector_bounding_boxes.setEnabled(False)

        elif text == "DetectTree" :

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

        elif text == "VHRTrees":

            self.checkBox_raster_binary.setVisible(False)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(False)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

            self.checkBox_raster_binary.setChecked(False)
            self.checkBox_raster_grayscale.setChecked(False)
            self.checkBox_vector_polygons.setChecked(False)
            self.checkBox_vector_bounding_boxes.setChecked(True)
            self.checkBox_vector_centroids.setChecked(False)

            self.checkBox_vector_bounding_boxes.setEnabled(False)

        elif text == "Custom ONNX Model":

            self.mGroupBox_Custom.setVisible(True)

            self.mGroupBox_result_types.setVisible(False)


    def _update_extent_options(self, text):

        self.mExtentGroupBox.setVisible(False)

        if text == "Custom extent":
            self.mExtentGroupBox.setVisible(True)
            
    def _result_types_are_selected(self):
        
        valid = False
        
        checkboxes = [
            self.checkBox_raster_binary
            , self.checkBox_raster_grayscale
            , self.checkBox_vector_polygons
            , self.checkBox_vector_bounding_boxes
            , self.checkBox_vector_centroids
        ]
        
        for checkbox in checkboxes:
            if checkbox.isVisible():
                valid = valid or checkbox.isChecked()
                
        return valid
    
    def _handle_clean_cache_folder(self):
        """Handle the clean cache folder button click."""
        
        output_dir = self.mQgsFileWidget.filePath()
        
        from .process.tree.interface.cachemanager import CacheManager
        cache_manager = CacheManager(project_path=output_dir)
        cache_manager.clean_cache_folder()

        from qgis.core import QgsApplication, QgsTask
        from qgis.core import Qgis

        def completed(task):

            # QtWidgets.QMessageBox.information(
            #     self, "Cache Cleaned", "Cache folder has been cleaned successfully."
            # )

            iface.messageBar().pushMessage(f"Cache Cleaned", f"Cache Folder Cleared for {output_dir}", level=Qgis.Success)

        # Create a few tasks
        task = QgsTask.fromFunction('Clean Cache Folder', cache_manager.clean_cache_folder(),
                                    on_finished=completed)
        QgsApplication.taskManager().addTask(task)

    def run_simple_inference(self):
        from qgis import processing
        processing.execAlgorithmDialog('TreeEyed:simple_inference')

class SettingsDialog(QtWidgets.QDialog, FORM_CLASS_SETTINGS_DIALOG):

    def __init__(self, parent=None):

        super(SettingsDialog, self).__init__(parent)
        # Set up the user interface from Designer.
        # After setupUI you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://doc.qt.io/qt-5/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)

        self.setWindowTitle("TreeEyed - Settings")

        # Read existing
        s = QgsSettings()
        plugin_name = "TreeEyed"
        model_dir = s.value(plugin_name + "/modelDir", os.getcwd())

        self.mQgsFileWidget_model_dir.setFilePath(model_dir)

        self.mQgsFileWidget_model_dir.fileChanged.connect(self._update_model_dir)

        self.pushButton_show.clicked.connect(self._handle_buttonShowFolder)

    def _update_model_dir(self, model_dir):

        print(model_dir)

        s = QgsSettings()
        plugin_name = "TreeEyed"
        s.setValue(plugin_name + "/modelDir", model_dir)

    def _handle_buttonShowFolder(self):

        filepath = self.mQgsFileWidget_model_dir.filePath()

        if os.path.isdir(filepath):
            QDesktopServices.openUrl(QUrl.fromLocalFile(filepath))

    

