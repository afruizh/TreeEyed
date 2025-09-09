import os
import subprocess
from threading import Thread
import importlib
import requests

from qgis.utils import iface

from qgis.PyQt import QtCore, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtGui import QCloseEvent
from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QTextBrowser

import sys

from qgis.core import (
  QgsSettings
  , QgsTask
  , QgsTaskManager
  , QgsApplication
  , QgsMessageLog
)

import qgis

import random
from time import sleep

from qgis.core import (
    QgsApplication, QgsTask, QgsMessageLog, Qgis
    )

from qgis.core import Qgis

from .qgis_utils import *

import pkg_resources

def get_installed_packages():
    return {dist.project_name.lower() for dist in pkg_resources.working_set}

import re

def normalize_package_name(req):
    # Strip version specifiers like >=, ==, <=, etc.
    return re.split(r'[<>=!~]+', req.strip())[0].lower()

import subprocess

def get_package_dependencies(package_name):
    try:
        
        if "==" in package_name:
            name, version = package_name.split("==")
        else:
            name, version = package_name, None

        print(f"https://pypi.org/pypi/{name}/json")
        response  = requests.get(f"https://pypi.org/pypi/{name}/json", timeout=10)    
        response.raise_for_status()
        data = response.json()

        if not version:
            version = data["info"].get("version")

        requires = data["info"].get("requires_dist", [])
        if not requires:
            return []

        return [r.split(";")[0].split()[0].lower() for r in requires if r]
    except Exception as e:
        print("Error:", e)
        return []

def get_missing_dependencies(package_name):
    installed = get_installed_packages()
    print("installed")
    print(installed)
    required = get_package_dependencies(package_name)
    required_clean = [normalize_package_name(pkg) for pkg in required]
    print("required")
    print(required)
    missing = [pkg for pkg in required_clean if pkg not in installed]
    print("missing")
    print(missing)
    return missing
    #return []



class InstallerManager():

    def __init__(self):

        self.plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.python_command = "python"
        self.install_dir = os.path.join(self.plugin_dir, "dependencies")
        self.install_dir = os.path.normpath(self.install_dir)

        self.task_manager = QgsApplication.taskManager() # Solve bug not running first time?

        # self.packages = [#'deepforest'
        #             'gdown==5.2.0'
        #             ,'rasterio==1.3.10' ## version 1.3.11 generates bug
        #             , 'pycocotools==2.0.8'
        #             , 'torch==2.5.1'
        #             , 'torchvision==0.20.1'
        #             #, 'opencv-python==4.10.0.84'
        #             , 'opencv-python-headless==4.11'
        #             , 'deepforest==1.4.1'
        #             , 'scikit-learn==1.6.1'
        #             ]
        
        
        
        # self.packages_import = [#'deepforest'
        #             'gdown'            
        #             ,'rasterio'
        #             , 'pycocotools'
        #             , 'torch'
        #             , 'torchvision'
        #             , 'cv2'
        #             , 'deepforest'
        #             , 'sklearn'
        #             ]

        self.packages = [
                        'gdown==5.2.0'
                        #'pyproj==3.6.1'
                        #,'pyarrow==15.0.2'
                        #,'geopandas==0.14.4'
                        #,'rasterio==1.4.3' #,'rasterio==1.3.9'
                        ,'opencv-python-headless==4.11.0.86'
                         #,'rasterio==1.3.10'
                         , 'onnxruntime-gpu==1.22.0'
                         , 'pycocotools==2.0.8'
                         #, 'deepforest==1.4.1'                         
                         #, 'torch==2.5.1'
                         #, 'torchvision==0.20.1'
                         ]
        self.packages_import = [
                                'gdown' 
                                #'pyproj'
                                #,'pyarrow'
                                #,'geopandas'
                                #,'rasterio'
                                ,'cv2'
                                #, 'rasterio'
                                , 'onnxruntime'
                                , 'pycocotools'
                                #, 'deepforest'
                                #, "torch"
                                #, "tochvision"
                                ]

        #gdown

        return
    
    def get_install_commands(self):

        cmds = []

        for package in self.packages:
            cmd = [self.python_command, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--prefer-binary", f'--target={self.install_dir}']
            #cmd = ["uv", "pip", "install", "--no-cache-dir", f'--target={self.install_dir}']
            
            if package in ["torch", "torchvision", 'torch==2.5.1', 'torchvision==0.20.1']:
                cmd.append("--index-url")
                cmd.append("https://download.pytorch.org/whl/cu118")

            # install with dependencies
            if package in ["onnxruntime-gpu==1.22.0", 'gdown==5.2.0']:
                cmd = [self.python_command, "-m", "pip", "install", "--no-cache-dir", "--prefer-binary", f'--target={self.install_dir}']
            
            cmd.append(package)

            cmds.append(cmd)

        return cmds
    
    # def install_packages(self):

    #     cmd = [self.python_command, "-m", "pip", "install", f'--target={self.install_dir}']
    #     #, f'--target={PACKAGES_INSTALL_DIR}'

    #     for package in self.packages:
    #         cmd.append(package)

    #     with subprocess.Popen(cmd, stdout = subprocess.PIPE) as proc:
    #         print(proc.stdout.read())

    def check_imports(self):

        if self.install_dir not in sys.path:
            sys.path.append(self.install_dir)  # TODO: check for a less intrusive way to do this

        

        packages_import_list = self.packages_import.copy()


        new_packages_import = []
        new_packages = []

        res = True

        for index, package in enumerate(packages_import_list):

            try:
                print(package)
                importlib.import_module(package)

                # self.packages_import.pop(count)
                # self.packages.pop(count)

            except Exception as e:

                new_packages_import.append(self.packages_import[index])
                new_packages.append(self.packages[index])

                print("Dependencies could not be imported")
                print(e)
                QgsMessageLog.logMessage("Dependencies could not be imported",MESSAGE_CATEGORY, Qgis.Critical)
                QgsMessageLog.logMessage(str(e),MESSAGE_CATEGORY, Qgis.Critical)
                #return False
                res = False


        
        self.packages = new_packages
        self.packages_import = new_packages_import

        print(self.packages)
        print(self.packages_import)

        return res

MESSAGE_CATEGORY = 'Tree Eyed Plugin'

class InstallerTask(QgsTask):
    finished_signal = pyqtSignal(bool)

    def __init__(self, description, installer_manager = None):
        super().__init__(description, QgsTask.CanCancel)

        self.installer_manager = installer_manager

    def run(self):

        self.setProgress(5)

        #self.setProgress(10)

        QgsMessageLog.logMessage('Started task "{}"'.format(
                                     self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        

        #, f'--target={PACKAGES_INSTALL_DIR}'
        if self.installer_manager is None:
            im = InstallerManager()
        else:
            im = self.installer_manager

        

        cmds = im.get_install_commands()

        for index,cmd in enumerate(cmds):

            QgsMessageLog.logMessage(str(cmd),MESSAGE_CATEGORY, Qgis.Info)

            if index == 0:
                self.setProgress(1)
            else:
                step_progress = (index)*1.0/len(cmds)*100
                self.setProgress(step_progress)

                QgsMessageLog.logMessage(str(step_progress),MESSAGE_CATEGORY, Qgis.Info)

            full_cmd = ' '.join(cmd)
            QgsMessageLog.logMessage("If you encounter any trouble please open OsGeoW4 Shell and run the following command \"" + full_cmd + "\"",MESSAGE_CATEGORY, Qgis.Info)


            # plugin_dir = os.path.dirname(im.install_dir)
            # cwd_path = os.path.join(plugin_dir, 'dependencies', 'bin')
            # cwd_path = os.path.normpath(cwd_path)
            # print(cwd_path)
        
            #with subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd_path) as proc:
            with subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                error_found = False
                for line in proc.stdout:
                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                    print(decoded_line)
                    #QgsMessageLog.logMessage(str(line),MESSAGE_CATEGORY, Qgis.Info)
                    QgsMessageLog.logMessage(decoded_line,MESSAGE_CATEGORY, Qgis.Info)

                    if any(keyword in decoded_line.lower() for keyword in ["error", "exception", "traceback", "failed"]):
                        if "pip's dependency resolver does not currently" not in decoded_line.lower():
                            error_found = True

                    if self.isCanceled():
                        return False

                    if error_found:
                        QgsMessageLog.logMessage("Error detected during installation.", MESSAGE_CATEGORY, Qgis.Critical)
                        return False
                    
            missing_deps = get_missing_dependencies(cmd[-1])
            for dep in missing_deps:

                print(dep)

                cmd_deps = ["python", "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--prefer-binary", f'--target={im.install_dir}']
                
                if cmd[-1] in ["torch", "torchvision", 'torch==2.5.1', 'torchvision==0.20.1']:
                    cmd_deps.append("--index-url")
                    cmd_deps.append("https://download.pytorch.org/whl/cu118")
                
                cmd_deps.append(dep)

                QgsMessageLog.logMessage(f"Installing dependencies for {cmd[-1]}",MESSAGE_CATEGORY, Qgis.Info)
                QgsMessageLog.logMessage(str(cmd_deps),MESSAGE_CATEGORY, Qgis.Info)
                full_cmd = ' '.join(cmd_deps)
                QgsMessageLog.logMessage("If you encounter any trouble please open OsGeoW4 Shell and run the following command \"" + full_cmd + "\"",MESSAGE_CATEGORY, Qgis.Info)

                with subprocess.Popen(cmd_deps, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
                    error_found = False
                    
                    for line in proc.stdout:
                        decoded_line = line.decode('utf-8', errors='replace').rstrip()
                        print(decoded_line)
                        #QgsMessageLog.logMessage(str(line),MESSAGE_CATEGORY, Qgis.Info)
                        QgsMessageLog.logMessage(decoded_line,MESSAGE_CATEGORY, Qgis.Info)

                        if any(keyword in decoded_line.lower() for keyword in ["error", "exception", "traceback", "failed"]):
                            if "pip's dependency resolver does not currently" not in decoded_line.lower():
                                error_found = True

                        if self.isCanceled():
                            return False

                        if error_found:
                            QgsMessageLog.logMessage("Error detected during installation.", MESSAGE_CATEGORY, Qgis.Critical)
                            return False


            if self.isCanceled():
                return False

        self.setProgress(100)

        self.finished_signal.emit(True)

        return True

    def finished(self, result):

        print("finished")

        # if result:
        #     QgsMessageLog.logMessage("Installation successful! \nPlease restart QGIS application to be able to use TreeEyed plugin.",MESSAGE_CATEGORY, Qgis.Success)
            
        #     msg = QMessageBox(iface.mainWindow())
        #     msg.setWindowTitle("Tree Eyed")
        #     msg.setWindowModality
        #     msg.setText("Installation successful! \nPlease restart QGIS application to be able to use TreeEyed plugin.")
        #     msg.setIcon(QMessageBox.Information)


        #     print("reloading")
        #     qgis.utils.reloadPlugin("tree_eyed")
        # else:
        #     # QgsMessageLog.logMessage(
        #     #         'RandomTask "{name}" Exception: {exception}'.format(
        #     #             name=self.description(),
        #     #             exception=self.exception),
        #     #         MESSAGE_CATEGORY, Qgis.Critical)
        #     QgsMessageLog.logMessage("Installation was not successful!",MESSAGE_CATEGORY, Qgis.Critical)


    def cancel(self):
        QgsMessageLog.logMessage('Package installation was canceled',MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()

def check_packages(iface):

    QgsMessageLog.logMessage("checking packages",MESSAGE_CATEGORY, Qgis.Info)

    im = InstallerManager()
    if im.check_imports():
        return True

    msg = QMessageBox(iface.mainWindow())
    msg.setWindowTitle("Tree Eyed")
    msg.setWindowModality
    msg.setText("Additional python packages are required to use this plugin.\nDo you want to install them? It may take a while.")
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
    ret = msg.exec()

    if ret == QMessageBox.Yes:

        # Open log messages
        qgis_utils_show_log_messages_panel()
        
        QgsMessageLog.logMessage("Installing additional packages", MESSAGE_CATEGORY, Qgis.Warning)
        
        # Run install
        installer_task = InstallerTask('Tree Eyed installing python packages', im)
        installer_task.finished_signal.connect(installer_finished)
        im.task_manager.addTask(installer_task)
        QgsMessageLog.logMessage("Installing additional packages started", MESSAGE_CATEGORY, Qgis.Warning)

        return False

    elif ret == QMessageBox.No:
        print("No was clicked")
        QgsMessageLog.logMessage("Installing additional packages canceled",MESSAGE_CATEGORY, Qgis.Warning)
        
        return False
    
def installer_finished(result):


    if result:
        QgsMessageLog.logMessage("Installation successful! \nPlease restart QGIS application to be able to use TreeEyed plugin.",MESSAGE_CATEGORY, Qgis.Success)

        msg = QMessageBox(qgis.utils.iface.mainWindow())
        msg.setWindowTitle("Tree Eyed")
        msg.setText("Installation successful! \n\nPlease restart QGIS application.")
        msg.setIcon(QMessageBox.Information)
        #msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
        ret = msg.exec()

        qgis.utils.reloadPlugin("tree_eyed")
    else:
        # QgsMessageLog.logMessage(
        #         'RandomTask "{name}" Exception: {exception}'.format(
        #             name=self.description(),
        #             exception=self.exception),
        #         MESSAGE_CATEGORY, Qgis.Critical)
        QgsMessageLog.logMessage("Installation was not successful!",MESSAGE_CATEGORY, Qgis.Critical)



