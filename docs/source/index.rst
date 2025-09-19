.. my_test_doc documentation master file, created by
   sphinx-quickstart on Thu Jul  4 08:22:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: res/logos/Banner_TreeEyed.png
   :alt: Plugin Installation

.. raw:: html

   <div align="center">
   
   <a href="https://github.com/afruizh/TreeEyed/releases/tag/v0.2.0">
      <img src="https://img.shields.io/badge/Download-Latest%20Release-brightgreen.svg?logo=github" alt="Download Latest Release">
   </a>
   <a href="https://treeeyed.readthedocs.io/en/latest/">
      <img src="https://img.shields.io/badge/Docs-Read%20The%20Docs-blue.svg?logo=readthedocs" alt="Documentation">
   </a>
   <a href="https://discord.gg/cXZhWgfqt2">
      <img src="https://img.shields.io/badge/Discord-Join%20Community-7289da.svg?logo=discord&logoColor=white" alt="Discord">
   </a>
   
   </div>

.. raw:: html

   <div align="center">
   
   <a href="https://python.org">
      <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
   </a>
   <a href="https://qgis.org">
      <img src="https://img.shields.io/badge/QGIS-Compatible-green.svg" alt="QGIS">
   </a>
   <a href="LICENSE">
      <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License">
   </a>
   <a href="https://onnxruntime.ai">
      <img src="https://img.shields.io/badge/ONNX-Runtime-red.svg" alt="ONNX">
   </a>
   
   </div>

Home
======

======================
TreeEyed QGIS Plugin
======================

**Version:** 0.2.0

TreeEyed is a QGIS plugin to leverage AI models for tree monitoring using remote sensing imagery.


.. raw:: html

   <div style="text-align:center;width:100%;">
      <iframe  width="560" height="315" src="https://www.youtube.com/embed/QnMAEX6qkGU?si=7wMHSbk9K2zPT-sY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
   </div>


==================
Features
==================
This plugin seeks to integrate existing and custom AI models for tree monitoring (semantic segmentation, instance segmentation, and object detection) in **high resolution RGB imagery**.

Apart from the model handling this plugin facilitates the integration with QGIS layers for image extraction and post-processing. Additional features for dataset creation and validation in `COCO format <https://cocodataset.org/#format-data>`_ are available.  

Available models:

* `HighResCanopyHeight <https://github.com/facebookresearch/HighResCanopyHeight>`_
* Custom Mask-RCNN
* `DeepForest <https://github.com/weecology/DeepForest>`_
* `VHRTrees <https://github.com/RSandAI/VHRTrees>`_

==================
Acknowledgment
==================

This work is being carried out as part of the CGIAR Initiatives and Funded Projects. CGIAR is a global research partnership for a food-secure future. Its science is carried out by 15 Research Centers in close collaboration with hundreds of partners across the globe.

TreeEyed is an open source project, and we welcome contributions and feedback from the community.  
We would like to extend our gratitude to the developers and maintainers of the libraries and models integrated into this plugin:

* `HighResCanopyHeight <https://github.com/facebookresearch/HighResCanopyHeight>`_
* `DeepForest <https://github.com/weecology/DeepForest>`_
* `VHRTrees <https://github.com/RSandAI/VHRTrees>`_

==================
Authors
==================

.. image:: res/logos/tf_small.png
   :alt: Plugin Installation

Tropical Forages Program

Alliance Bioversity International & CIAT

==================
Research
==================

   A. F. Ruiz-Hurtado, J. P. Bolaños, D. A. Arrechea-Castillo, and J. A. Cardoso, ‘TreeEyed: A QGIS plugin for tree monitoring in silvopastoral systems using state of the art AI models’, SoftwareX, vol. 29, p. 102071, Feb. 2025, `doi: 10.1016/j.softx.2025.102071 <https://www.sciencedirect.com/science/article/pii/S235271102500038X>`_.

   Citation (Bibtex format):

.. code-block:: RST

   @article{ruiz-hurtadoTreeEyedQGISPlugin2025,
   title = {{{TreeEyed}}: {{A QGIS}} Plugin for Tree Monitoring in Silvopastoral Systems Using State of the Art {{AI}} Models},
   author = {{Ruiz-Hurtado}, Andres Felipe and Bola{\~n}os, Juliana Perez and {Arrechea-Castillo}, Darwin Alexis and Cardoso, Juan Andres},
   year = {2025},
   month = feb,
   journal = {SoftwareX},
   volume = {29},
   pages = {102071},
   issn = {2352-7110},
   doi = {10.1016/j.softx.2025.102071},
   keywords = {Computer vision,Deep learning,QGIS,Remote sensing,Silvopastoral systems,Tree monitoring},
   }



----

==================
References
==================



==================
Contents
==================

.. toctree::
   :maxdepth: 2

   self

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/page_installation
   pages/page_download_models
   pages/page_features

.. toctree::
   :maxdepth: 2
   :caption: QuickStart

   examples/example_simple_analysis
   examples/tips

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/example_1
   examples/example_2
   examples/example_3

.. toctree::
   :maxdepth: 2
   :caption: Developers

   developers/developers


.. .. toctree::
..    :maxdepth: 2
..    :caption: User Guide

..    usage
..    examples

.. .. toctree::
..    :maxdepth: 2
..    :caption: Developer Guide

..    contributing
..    testing
