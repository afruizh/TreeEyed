import os
import hashlib

class CacheManager():

    def __init__(self, project_path=None, cache_dir_name=".cache_tree_eyed", temp_raster_name="_tree_eyed_temp_raster.tif", parameters = None):

        self.cache_dir_name = cache_dir_name 
        self.temp_raster_name = temp_raster_name
        self.parameters = parameters

        if project_path:
            self.base_path = os.path.join(project_path, self.cache_dir_name)
            self.base_path = os.path.normpath(self.base_path)
        else:
            from qgis.PyQt.QtCore import QStandardPaths
            self.base_path = os.path.join(
                QStandardPaths.writableLocation(QStandardPaths.CacheLocation),
                "TreeEyed"
            )
            self.base_path = os.path.normpath(self.base_path)

    def compute_key(self, *args):
        return hashlib.md5("_".join(map(str, args)).encode()).hexdigest()

    def get_cache_folder_path(self, stage, key):
        folder = os.path.join(self.base_path, stage, key)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    def clean_cache_folder_path(self, stage, key):

        folder = self.get_cache_folder_path(stage, key)
        if os.path.exists(folder):
            for root, dirs, files in os.walk(folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(folder)

        return True

    def exists(self, stage, key):
        return os.path.exists(self.get_cache_path(stage, key))
    
    def clean_cache_folder(self):

        if os.path.exists(self.base_path):
            for root, dirs, files in os.walk(self.base_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.base_path)

        return True
    
    def get_cache_folder(self):
        os.makedirs(self.base_path, exist_ok=True)
        return self.base_path
    
    def get_temp_raster_path(self):
        return os.path.join(self.base_path, self.temp_raster_name)
    
    def remove_temp_raster(self):
        temp_raster_path = self.temp_raster_path()
        if os.path.exists(temp_raster_path):
            os.remove(temp_raster_path)
        return True

