from qgis.PyQt.QtCore import QTimer
from qgis.core import QgsApplication
from qgis.utils import plugins

def reload_plugin_safely(plugin_name):
    """Safely reloads the plugin using QGIS internals."""
    plugin_mgr = QgsApplication.instance().pluginManagerInterface()

    if plugin_name not in plugins:
        print(f"Plugin '{plugin_name}' not currently loaded.")
        return

    # Unload the plugin
    try:
        print(f"Unloading plugin: {plugin_name}")
        plugin_mgr.unloadPlugin(plugin_name)
    except Exception as e:
        print(f"Error unloading plugin: {e}")
        return

    # Delay reload to let cleanup settle
    def _reload():
        try:
            print(f"Reloading plugin: {plugin_name}")
            plugin_mgr.loadPlugin(plugin_name)
            plugin_mgr.startPlugin(plugin_name)
            print(f"Plugin '{plugin_name}' reloaded successfully.")
        except Exception as e:
            print(f"Error reloading plugin: {e}")

    # Use a short delay to let the event loop process the unload
    QTimer.singleShot(500, _reload)
	
	
def on_task_finished():
    print("Task completed. Reloading plugin to reset DLL state.")
    reload_plugin_safely("your_plugin_folder_name")