# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
# Initialize Qt resources from file resources.py
from .resources import *

from qgis.core import QgsApplication

# Import the code for the DockWidget
from .tree_eyed_dockwidget import TreeEyedDockWidget
from .tree_eyed_provider import TreeEyedProvider
import os.path

import os


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



class TreeEyed:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        self.provider = None
        # Save reference to the QGIS interface
        self.iface = iface

        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)

        # # update default paths
        # global DEFAULT_RASTER_COLORMAP
        # DEFAULT_RASTER_COLORMAP = os.path.join(self.plugin_dir, "colormap.txt")
        # global DEFAULT_VECTOR_BB
        # DEFAULT_VECTOR_BB = os.path.join(self.plugin_dir, "detection_style.qml")

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

        self.tree_eyed_processor = None


    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        self.provider = TreeEyedProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)


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
        self.initProcessing()

        # Add toolbar button for the pluing
        icon_path = ':/plugins/tree_eyed/res/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Tree Eyed'),
            callback=self.run,
            parent=self.iface.mainWindow())
        
        # # Add toolbar butotn for the processing provider
        # icon_path = ':/plugins/tree_eyed/res/icon_black.png'
        # icon = QIcon(icon_path)
        # action = QAction(icon, "Simple Inference", self.iface.mainWindow())
        # action.triggered.connect(self.run_simple_inference)
        # self.toolbar.addAction(action)

        icon_path = ':/plugins/tree_eyed/res/icon_black.png'
        self.add_action(
            icon_path,
            text=self.tr(u"Simple Inference"),
            callback=self.run_simple_inference,
            parent=self.iface.mainWindow()
        )

        # Add Settings action to the menu
        icon_path = ':/plugins/tree_eyed/res/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u"Settings"),
            callback=self.open_settings_dialog,
            add_to_toolbar=False,
            parent=self.iface.mainWindow()
        )

    def open_settings_dialog(self):
        """Open the settings dialog from the dockwidget."""
        # If dockwidget exists and has settings dialog, use it; otherwise, create a temporary one
        from .tree_eyed_dockwidget import TreeEyedDockWidget
        if self.dockwidget is not None:
            self.dockwidget._open_settings()
        else:
            self.run()  # Ensure the dockwidget is created
            self.dockwidget._open_settings()

        

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
        QgsApplication.processingRegistry().removeProvider(self.provider)

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

            #Checks go here


            #print "** STARTING TreeEyed"

            # dockwidget may not exist if:
            #    first run of plugin
            #    removed on close (see self.onClosePlugin method)
            if self.dockwidget == None:
                # Create the dockwidget (after translation) and keep reference
                self.dockwidget = TreeEyedDockWidget()

            # connect to provide cleanup on closing of dockwidget
            self.dockwidget.closingPlugin.connect(self.onClosePlugin)

            # Check if install packages
            from .process.gui_utils import installer
            res = installer.check_packages(self.iface)
            #res = True

            if not res:
                self.pluginIsActive = False
                return
            else:
                # REMOVED: Causes PyArrow/GeoPandas threading crashes
                # from .process.tree.custom_processor import  initialize_thread_safe_environment
                # initialize_thread_safe_environment()

                #Load 
                from .tree_eyed_processor import TreeEyedProcessor
                self.tree_eyed_processor = TreeEyedProcessor(self.iface)

                # Add cuda paths in setting to environment path
                from qgis.core import QgsSettings  
                plugin_name = "TreeEyed"
                dirs = QgsSettings().value(plugin_name + "/cudaDirs", [])
                sep = os.pathsep
                current_path = os.environ.get("PATH", "")
                if current_path:
                    new_path = sep.join(dirs) + sep + current_path
                else:
                    new_path = sep.join(dirs)
                os.environ["PATH"] = new_path
                # add to dll directory
                for dir in dirs:
                    os.add_dll_directory(dir)
                print("PATH:", os.environ["PATH"])

                
                
                # Additional Connections
                self.dockwidget.process_signal.connect(self.tree_eyed_processor._process)
                self.dockwidget.download_models_signal.connect(self.tree_eyed_processor._prompt_download_models)

            # Additional Connections
            self.iface.mapCanvas().scaleChanged.connect(self.dockwidget._handle_mapScaleChanged)  
            
            # show the dockwidget            
            # TODO: fix to allow choice of dock location
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dockwidget)
            self.dockwidget.show()

    #--------------------------------------------------------------------------
    # Additional functions

    def run_simple_inference(self):
        from qgis import processing
        processing.execAlgorithmDialog('TreeEyed:simple_inference')


    