from qgis.utils import iface
from .config import *
from qgis.PyQt.QtWidgets import QDockWidget, QTabWidget, QWidget

def qgis_utils_get_layer_dims(layer):

    extent = layer.extent()
    data_provider = layer.dataProvider()
    
    xRes = -1
    yRes = -1

    # if (data_provider.xSize() > 0 and data_provider.ySize() > 0):
    #     xRes = extent.width() / data_provider.xSize()
    #     yRes = extent.height() / data_provider.ySize()
    
    xRes = data_provider.xSize()
    yRes = data_provider.ySize()
        
    return xRes, yRes

def qgis_utils_get_current_mapview_dims():
    
    extent = iface.mapCanvas().extent()
    width = iface.mapCanvas().size().width()
    height = iface.mapCanvas().size().height()
    
    config_debug("mapview", width, height)
    
    xRes = -1
    yRes = -1

    # if (width > 0 and height > 0):

    #     xRes = extent.width() / width
    #     yRes = extent.height() / height
    
    xRes = width
    yRes = height
        
    return xRes, yRes

def qgis_utils_valid_dims(xRes, yRes):
    
    if xRes <= 0 or yRes <= 0:
        return False
    
    tiles = int(xRes/256)*int(yRes/256)
    config_debug(tiles)
    
    if tiles > CONFIG_MAX_TILE_PROCESSING:
        return False
    
    return True
    
def qgis_utils_show_log_messages_panel(show = True, tab_title = "Tree Eyed Plugin"):
    """
    Shows and raises the Log Messages Panel in QGIS, and optionally switches 
    to a specific tab if it exists.
    """

    for x in iface.mainWindow().findChildren(QDockWidget):
        if x.objectName() == 'MessageLog':
            x.setVisible(show)
            if show:
                x.raise_()
                for tabwidget in x.findChildren(QTabWidget):
                    #print(tabwidget.objectName())
                    for index in range(tabwidget.count()):
                        #print(tabwidget.tabText(index))
                        if (tabwidget.tabText(index) == tab_title):
                            tabwidget.setCurrentIndex(index)
