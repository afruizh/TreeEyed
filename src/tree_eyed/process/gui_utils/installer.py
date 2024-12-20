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

class InstallerManager():

    def __init__(self):

        self.plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.python_command = "python"
        self.install_dir = os.path.join(self.plugin_dir, "dependencies")

        self.task_manager = QgsApplication.taskManager() # Solve bug not running first time?

        self.packages = [#'deepforest'
                    'gdown'
                    ,'rasterio==1.3.10' ## version 1.3.11 generates bug
                    , 'pycocotools'
                    , 'torch'
                    , 'torchvision'
                    , 'opencv-python==4.10.0'
                    , 'deepforest'
                    , 'scikit-learn'
                    ]
        
        self.packages_import = [#'deepforest'
                    'gdown'            
                    ,'rasterio'
                    , 'pycocotools'
                    , 'torch'
                    , 'torchvision'
                    , 'cv2'
                    , 'deepforest'
                    , 'sklearn'
                    ]

        #self.packages = ["pycocotools"]
        #self.packages_import = ["pycocotools"]

        return
    
    def get_install_commands(self):

        cmds = []

        for package in self.packages:
            cmd = [self.python_command, "-m", "pip", "install", f'--target={self.install_dir}']
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

            step_progress = (index)*1.0/len(cmds)*100
            self.setProgress(step_progress)

            QgsMessageLog.logMessage(str(step_progress),MESSAGE_CATEGORY, Qgis.Info)
        
            with subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE) as proc:
                for line in proc.stdout:

                    print(line)
                    QgsMessageLog.logMessage(str(line),MESSAGE_CATEGORY, Qgis.Info)


                    if self.isCanceled():
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



