from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon
import os

from .tree_eyed_inference_algorithm import TreeEyedInferenceAlgorithm

class TreeEyedProvider(QgsProcessingProvider):
    """Processing provider for TreeEyed plugin."""

    def __init__(self):
        """
        Default constructor.
        """
        QgsProcessingProvider.__init__(self)

    def unload(self):
        """
        Unloads the provider. Any tear-down steps required by the provider
        should be implemented here.
        """
        pass

    def loadAlgorithms(self):
        """
        Loads all algorithms belonging to this provider.
        """
        self.addAlgorithm(TreeEyedInferenceAlgorithm())


    def id(self):
        """
        Returns the unique provider id, used for identifying the provider. This
        string should be a unique, short, character only string, eg "qgis" or
        "gdal". This string should not be localised.
        """
        return 'TreeEyed'

    def name(self):
        """
        Returns the provider name, which is used to describe the provider
        within the GUI.

        This string should be short (e.g. "Lastools") and localised.
        """
        return self.tr('TreeEyed')

    def icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), 'res/icon.png')
        return QIcon(icon_path)

    def longName(self):
        """
        Returns the a longer version of the provider name, which can include
        extra details such as version numbers. E.g. "Lastools LIDAR tools
        (version 2.2.1)". This string should be localised. The default
        implementation returns the same string as name().
        """
        return self.name()
