Inference using DeepForest model
===================================

This example uses a raster file from the original DeepForest Respository.
For additional information about the model performance please refer to the repository.

**DeepForest** is an **Object Detection Model**. That means it will generate a bounding box of each detected object.
That means that for this case 2 vector layers can be generated using TreeEyed:

Direct result:

* Bounding Boxes vector layer

Derived results:

* Centroids vector layer

===================
Input
===================

For this example we will use this image of a silvopastoral system.

.. csv-table:: Input image features
   :file: ../res/example_3.csv
   :align: center
   :header-rows: 1

.. image:: ../res/input_raster_png/example_3.png

The raster file is available to download `here <https://github.com/afruizh/TreeEyed/blob/documentation/examples/images/input_raster/example_3.tif>`_.

===================
Configuration
===================

Input
------

* Select the appropiate **Input layer** from the dropdown menu.
* For **Extent** select **Layer extent**.

Output
-------
* Select the appropiate **Output directory** and **Output name**.

Processing
----------

* Go to the **Inference** tab
   * Select **DeepForest** model from the dropdown menu   
   * Select the desired **Result types**, the direct result type for this model are **Bounding Boxes**
   * Press **Process** to start the inference process


===================
Results
===================

The resulting added layers depend on the selected **Result types**

.. .. list-table::
..    :widths: 25 25 25 25

..    * - .. image:: ../res/results_png/example_1_grayscale.png
..      - .. image:: ../res/2-3-2023-crop2.png
..      - .. image:: ../res/2-3-2023-crop3.png
..      - .. image:: ../res/2-3-2023-crop4.png

Direct result:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Processing Result
     - Result Type
     - Result
   * - **RASTER LAYERS**
     -
     -
   * - Derived Result
     - Binary
     - NOT AVAILABLE FOR THIS MODEL
   * - Derived result
     - Grayscale
     - NOT AVAILABLE FOR THIS MODEL
   * - **VECTOR LAYERS**
     -
     -  
   * - Derived result
     - Polygons
     - NOT AVAILABLE FOR THIS MODEL  
   * - **Direct result**
     - Bounding Boxes
     - .. image:: ../res/results_png/example_3_bb.png
   * - Derived result
     - Centroids
     - .. image:: ../res/results_png/example_3_centroids.png 