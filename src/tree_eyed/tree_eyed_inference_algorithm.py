
from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsFeatureSink,
                       QgsProcessingAlgorithm,
                       QgsProcessingParameterFeatureSource,
                       QgsProcessingParameterFeatureSink)

from qgis.core import QgsProcessingParameterRasterLayer, QgsProcessingParameterFileDestination
import subprocess
import tempfile
from qgis.core import QgsVectorLayer, QgsProject
from qgis.core import QgsProcessingParameterEnum

class TreeEyedInferenceAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is not need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    OUTPUT = 'OUTPUT'
    INPUT = 'INPUT'
    MODEL = 'MODEL'

    def initAlgorithm(self, config):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # Change input to raster layer selector
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                self.tr('Input raster layer')
            )
        )

        model_options = ['HighResCanopyHeight', 'Mask R-CNN', 'DeepForest', 'VHRTrees']
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL,
                self.tr('Model to use'),
                options=model_options,
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                self.tr('Output file (extension depends on model and will be corrected automatically)'),
                'GeoTIFF (*.tif);;ESRI Shapefile (*.shp)'
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        # Get parameters
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        #raster_path = raster_layer.source()
        output_filepath = self.parameterAsFileOutput(parameters, self.OUTPUT, context)

        model_index = self.parameterAsEnum(parameters, self.MODEL, context)
        model_name = ['HighResCanopyHeight', 'Mask R-CNN', 'DeepForest', 'VHRTrees'][model_index]

        outputs_defaults = {
            'HighResCanopyHeight':{
                "raster_outputs":["grayscale"]
                , "vector_outputs": []
                }
            ,'Mask R-CNN':{
                "raster_outputs":["binary"]
                , "vector_outputs": []
                }
            ,'DeepForest':{
                "raster_outputs":[]
                , "vector_outputs": ["bounding_boxes"]
                }
            ,'VHRTrees':{
                "raster_outputs":[]
                , "vector_outputs": ["bounding_boxes"]
                }

        }

        # *********************************
        from qgis.core import QgsProject
        from qgis.core import QgsCoordinateTransform
        from qgis.core import QgsCoordinateReferenceSystem

        layer = raster_layer

        extent = layer.extent()
        epgs = layer.crs().authid()
        
        layer_path = layer.dataProvider().dataSourceUri()
        input_raster_path = layer_path

        crs_source = layer.crs()  # The source CRS of the layer
        crs_destination = QgsCoordinateReferenceSystem("EPSG:4326")  # The destination CRS
        transform = QgsCoordinateTransform(crs_source, crs_destination, QgsProject.instance())

        # Transform the extent to EPSG:4326
        extent_transformed = transform.transformBoundingBox(extent)

        extent = extent_transformed

        from qgis.core import QgsSettings
        import os

        s = QgsSettings()
        plugin_name = "TreeEyed"
        model_dir = s.value(plugin_name + "/modelDir", "NODATA")
        model_dir = os.path.normpath(model_dir)

        from .process.tree.interface.cachemanager import CacheManager
        
        #cache_manager = CacheManager()
        #output_path = cache_manager.base_path

        output_path = os.path.dirname(output_filepath)

        prefix = os.path.basename(output_filepath)
        # remove extension if any
        if '.' in prefix:
            prefix = prefix.split('.')[0]

        task_parameters = {
            "model": model_name
            , "model_dir": model_dir
            , "output_path": output_path
            , "prefix": prefix
            #, "input_raster_path": raster_path
            #, "hrch_type": "Satellite"
            #, "hrch_threshold": 0.15
            #, "layer": <QgsRasterLayer: "HSJ_4_77" (gdal)>
            #, "raster2vector_layer": <QgsRasterLayer: "HSJ_4_77" (gdal)>
            #, "raster2vector_threshold": 15
            #, "filter_area_layer": <QgsVectorLayer: "results99_vector" (ogr)>
            #, "filter_area_area": 100.0
            , "task": "inference"
            #, "raster_outputs": ["binary"]
            #, "vector_outputs": ["polygons"]
            , "extent_type": "Layer extent"
            #, "extent": "<QgsRectangle: 0 0, 0 0>"
            #, "extent_crs": <QgsCoordinateReferenceSystem: EPSG:4326>
            #, "input_image": <QgsRasterLayer: "HSJ_4_77" (gdal)>
            #, "annotations": <QgsVectorLayer: "results99_vector" (ogr)>
            #, "num_tiles": 1024
            #, "validate_ground_truth": ""
            #, "validate_prediction": ""
            #, "output_format": "PNG"
            #, "overlap": 0
            #, "tile_size": 1024
            #, "force_tiling": False
            #, "epsg":""
        }

        
        
        # fix necessary 
        task_parameters['task'] = "inference"
        task_parameters['input_raster_path'] = input_raster_path

        minx = extent.xMinimum()
        miny = extent.yMinimum()
        maxx = extent.xMaximum()
        maxy = extent.yMaximum()
        bounds = (minx, miny, maxx, maxy)

        task_parameters["extent_general"] = bounds
        task_parameters['epsg'] = epgs

        task_parameters["output_files"] = []

        task_parameters.update(outputs_defaults[model_name])

        print(task_parameters)

        # Run a qgstask to handle the inference process
        from .tree_eyed_processor import WorkerTask
        from qgis.core import QgsApplication

        from .tree_eyed_processor import TreeEyedProcessor
        from qgis.utils import iface
        tree_eyed_processor = TreeEyedProcessor(iface)

        # qgstask = WorkerTask("Tree inference task", parameters)
        # qgstask.task_finished.connect(tree_eyed_processor._process_task_finished)
        # QgsApplication.taskManager().addTask(qgstask)  

        def custom_progress_callback(info):
            progress = info["progress"]*90 # use 90 to avoid reaching 100% before the end
            feedback.setProgress(progress)

        def custom_interruption_check():
            return feedback.isCanceled()

        from .process.tree.interface.processor import Processor
        processor = Processor(task_parameters
                                , progress_callback= custom_progress_callback
                                , interruption_check= custom_interruption_check)
        
        results = processor.run()

        print("***RESULT")
        print(results)

        if results["status"] == "error":
            

            #QgsMessageLog.logMessage(results["status"]+ " " + results["message"],MESSAGE_CATEGORY, Qgis.Critical)
            return False
        
        tree_eyed_processor._process_task_finished(results)

        # if not os.path.exists(output_path):

        #     plugin_dir = os.path.dirname(os.path.abspath(__file__))
        #     cwd_path = os.path.join(plugin_dir, 'ForagesROIs')
        #     exe_path = os.path.join(plugin_dir, 'ForagesROIs', 'ForagesROIs.exe')

        #     # Prepare environment with cwd_path added to PATH
        #     env = os.environ.copy()
        #     env["PATH"] = cwd_path# + os.pathsep + env.get("PATH", "")
        #     #proj_lib_path = r"C:\Program Files\QGIS 3.34.2\share\proj"  # Adjust if your QGIS is in a different folder
        #     #env["PROJ_LIB"] = cwd_path

        #     # Add cwd_path to the Python process PATH if not already present
        #     if cwd_path not in os.environ["PATH"]:
        #         os.environ["PATH"] = cwd_path# + os.pathsep + os.environ.get("PATH", "")
        #     #os.environ["PROJ_LIB"] = cwd_path

        #     #cmd = [exe_path, '--cli', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
        #     cmd = ['ForagesROIs.exe', '--cli', '--input', os.path.normpath(raster_path), '--output', os.path.normpath(output_path)]
        #     feedback.pushInfo(f"Process working directory (cwd): {cwd_path}")
        #     feedback.pushInfo(f"Process PATH: {env['PATH']}")
        #     #feedback.pushInfo(f"PROJ_LIB: {env['PROJ_LIB']}")
        #     feedback.pushInfo(f'Running: {" ".join(cmd)}')
        #     feedback.setProgress(10)  # Set progress to 10% before running the subprocess
        #     result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd_path, env=env, encoding='latin1')
        #     feedback.setProgress(80)  # Set progress to 80% after subprocess completes
            
        #     feedback.pushInfo(result.stdout)
            
        #     # Always show stderr if present
        #     if result.stderr:
        #         feedback.reportError(result.stderr)
        #         feedback.pushInfo(f"STDERR: {result.stderr}")

        #     if result.returncode != 0:
        #         raise Exception(f'Error running ForagesROIs.exe: {result.stderr}')

            

        # feedback.pushInfo("Loading output shapefile into QGIS...")

        # # Load the output shapefile into QGIS
        # if os.path.exists(output_path):
        #     if context.willLoadLayerOnCompletion(output_path):
        #         style_path = os.path.join(os.path.dirname(__file__), "ForagesROIs_style.qml")
        #         def apply_style(layer):
        #             if os.path.exists(style_path):
        #                 layer.loadNamedStyle(style_path)
        #                 layer.triggerRepaint()
        #             else:
        #                 feedback.pushInfo(f"Style file not found: {style_path}")
        #         context.addLayerToLoadOnCompletion(
        #             output_path,
        #             {
        #                 'layerName': os.path.basename(output_path),
        #                 'provider': 'ogr',
        #                 'postProcessor': apply_style
        #             }
        #         )
        #         feedback.pushInfo(f"Layer scheduled to load: {output_path}")
        #     else:
        #         feedback.pushInfo("Context will not load layer on completion.")

        #         # Manually load the layer into the project
        #         layer = QgsVectorLayer(output_path, os.path.basename(output_path), "ogr")
        #         if layer.isValid():
        #             QgsProject.instance().addMapLayer(layer)
        #             feedback.pushInfo(f"Layer loaded manually: {output_path}")
        #             # Optionally apply style
        #             style_path = os.path.join(os.path.dirname(__file__), "ForagesROIs_style.qml")
        #             if os.path.exists(style_path):
        #                 layer.loadNamedStyle(style_path)
        #                 layer.triggerRepaint()
        #         else:
        #             feedback.reportError(f"Failed to load layer manually: {output_path}")
        # else:
        #     feedback.reportError(f"Output file does not exist: {output_path}")

        return {self.OUTPUT: output_path}

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'simple_inference'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr(self.name())

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr(self.groupId())

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'TreeEyed'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return TreeEyedInferenceAlgorithm()
    
    def shortHelpString(self):
        return """        
        <p>This algorithm runs inference for the whole input raster layer using the selected model.</p>
        <h2>Input parameters</h2>
        <ul>
        <li><b>Input raster layer</b>: The raster to process.</li>
        <li><b>Model</b>: Select the model for inference. </li>
        <h2>Outputs</h2>
        <ul>
        <li><b>Output file</b>: Specify the output file path. For some models, output is a raster; for others, a shapefile.
            <ul>
                <li>HighResCanopyHeight, Mask R-CNN: <b>.tif</b> (GeoTIFF raster)</li>
                <li>DeepForest, VHRTrees: <b>.shp</b> (ESRI Shapefile)</li>
            </ul>
        </li>
        </ul>
        """


