.. my_test_doc documentation master file, created by
   sphinx-quickstart on Thu Jul  4 08:22:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TreeEyed Documentation
=======================================

TreeEyed is a QGIS plugin to leverage AI models for tree monitoring using remote sensing imagery.

.. raw:: html

   <div style="text-align:center;width:100%;">
      <iframe  width="560" height="315" src="https://www.youtube.com/embed/QnMAEX6qkGU?si=7wMHSbk9K2zPT-sY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
   </div>

==================
Features
==================
This plugins seeks to integrate existing and custom AI models for tree monitoring (semantic segmentation, instance segmentation, and object detection) in **high resolution RGB imagery**.

Apart from the model handling this plugin facilitates the integration with QGIS layers for image extraction and post-processing. Additional features for dataset creation and validation in COCO format are available.  

.. .. toctree::
..    :maxdepth: 2
..    :caption: Contents:

.. .. toctree::
..    :maxdepth: 1
..    :caption: Basics

..    pages/page_installation
..    pages/page_installation
   
   

.. .. toctree::
..    :maxdepth: 1
..    :caption: Examples

..    examples/example_simple_analysis


==================
Contents
==================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/page_installation
   pages/page_download_models

.. toctree::
   :maxdepth: 2
   :caption: QuickStart

   examples/example_simple_analysis

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
