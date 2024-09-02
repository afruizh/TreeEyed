import os
import subprocess
from threading import Thread
import importlib
import requests

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

class InstallerManager():

    def __init__(self):

        self.plugin_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.python_command = "python"
        self.install_dir = os.path.join(self.plugin_dir, "dependencies")

        self.packages = [#'deepforest'
                    'gdown'
                    ,'rasterio'
                    , 'pycocotools'
                    , 'torch'
                    , 'torchvision'
                    , 'opencv-python'
                    , 'deepforest'
                    ]
        
        self.packages_import = [#'deepforest'
                    'gdown'            
                    ,'rasterio'
                    , 'pycocotools'
                    , 'torch'
                    , 'torchvision'
                    , 'cv2'
                    , 'deepforest'
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

        try:

            for package in self.packages_import:
                importlib.import_module(package)

        except Exception as e:
            #print("Dependencies could not be imported")
            #print(e)
            QgsMessageLog.logMessage("Dependencies could not be imported",MESSAGE_CATEGORY, Qgis.Critical)
            QgsMessageLog.logMessage(str(e),MESSAGE_CATEGORY, Qgis.Critical)
            return False
        
        return True

MESSAGE_CATEGORY = 'Tree Eyed Plugin'

class InstallerTask(QgsTask):

    def __init__(self, description):
        super().__init__(description, QgsTask.CanCancel)

    def run(self):

        #self.setProgress(10)

        QgsMessageLog.logMessage('Started task "{}"'.format(
                                     self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        

        #, f'--target={PACKAGES_INSTALL_DIR}'
        im = InstallerManager()
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

        return True

    def finished(self, result):

        if result:
            QgsMessageLog.logMessage("Installation successful!",MESSAGE_CATEGORY, Qgis.Success)
            print("reloading")
            qgis.utils.reloadPlugin("tree_eyed")
        else:
            # QgsMessageLog.logMessage(
            #         'RandomTask "{name}" Exception: {exception}'.format(
            #             name=self.description(),
            #             exception=self.exception),
            #         MESSAGE_CATEGORY, Qgis.Critical)
            QgsMessageLog.logMessage("Installation was not successful!",MESSAGE_CATEGORY, Qgis.Critical)


    def cancel(self):
        QgsMessageLog.logMessage('Package installation was canceled',MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()

def check_packages(iface):

    QgsMessageLog.logMessage("checking packages",MESSAGE_CATEGORY, Qgis.Warning)

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
        
        QgsMessageLog.logMessage("Installing additional packages", MESSAGE_CATEGORY, Qgis.Warning)
        
        # Run install
        installer_task = InstallerTask('Tree Eyed installing python packages')
        QgsApplication.taskManager().addTask(installer_task)
        QgsMessageLog.logMessage("Installing additional packages started", MESSAGE_CATEGORY, Qgis.Warning)

        return False

    elif ret == QMessageBox.No:
        print("No was clicked")
        QgsMessageLog.logMessage("Installing additional packages canceled",MESSAGE_CATEGORY, Qgis.Warning)
        
        return False