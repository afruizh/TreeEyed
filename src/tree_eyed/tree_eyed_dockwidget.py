# -*- coding: utf-8 -*-

import os

from qgis.PyQt import QtGui, QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal, Qt

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

from .process.gui_utils.config import *

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'tree_eyed_dockwidget_base.ui'))

FORM_CLASS_SETTINGS_DIALOG, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'tree_eyed_settings_dialog.ui'))

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
        pixmap = QPixmap(":/plugins/tree_eyed/logo.png").scaledToWidth(200, Qt.SmoothTransformation)  # Scale to desired width
        self.label_6.setPixmap(pixmap)
        self.label_6.setAlignment(Qt.AlignCenter)
        
        pixmap = QPixmap(":/plugins/tree_eyed/forest.png").scaledToWidth(40, Qt.SmoothTransformation)  # Scale to desired width
        self.label_5.setPixmap(pixmap)
        self.label_5.setAlignment(Qt.AlignCenter)
        

        # Configure combobox filters
        self.mMapLayerComboBox_inputLayer.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBox_input_image.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mMapLayerComboBox_raster2vector.setFilters(QgsMapLayerProxyModel.RasterLayer)
        
        self.mMapLayerComboBox_annotations.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.mMapLayerComboBox_filter_area.setFilters(QgsMapLayerProxyModel.VectorLayer)
        
        self.pushButton_test_process.setVisible(False)
        self.mMapLayerComboBox_validation_ground_truth.setVisible(False)
        self.mMapLayerComboBox_validation_prediction.setVisible(False)
        

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


        # Map canvas collection
        print(self.parent)

    def _initial_configuration(self):
        self.mGroupBox_HRCH.setVisible(False)
        self.mExtentGroupBox.setVisible(False)

    def _open_settings(self):
        self.m_settings_dialog.open()

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
        }
       
        self.process_signal.emit(parameters)

    def _update_model_options(self, text):

        print(text)

        self.mGroupBox_HRCH.setVisible(False)

        if text == "HighResCanopyHeight":
            self.mGroupBox_HRCH.setVisible(True)

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(True)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

        elif text == "Mask R-CNN":

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

        elif text == "DeepForest":

            self.checkBox_raster_binary.setVisible(False)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(False)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

        elif text == "DetectTree" :

            self.checkBox_raster_binary.setVisible(True)
            self.checkBox_raster_grayscale.setVisible(False)
            self.checkBox_vector_polygons.setVisible(True)
            self.checkBox_vector_bounding_boxes.setVisible(True)
            self.checkBox_vector_centroids.setVisible(True)

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

    def _update_model_dir(self, model_dir):

        print(model_dir)

        s = QgsSettings()
        plugin_name = "TreeEyed"
        s.setValue(plugin_name + "/modelDir", model_dir)