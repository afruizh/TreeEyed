Tips and Tricks
===================================

* This plugin is intended for **High resoluton RGB images** make sure your input data is compliant.
* The gereferenciation is generated internally using the input data metadata. If the geodata is not correct the results may not be what is expected.
* You can use the area filtering option in **Post-process** to eliminate vector polygons that surpass an area value.
* You can check plugin's messages directly on the  **Log Messages Panel** (*View -> Panels -> Log Messages Panel*)
* Given the potential of concept drift, it is recommended to test models using different spatial resolutions closer to the each model's intented resolution.

================
Considerations
================

* Currently processing could take a long time, the plugin can handle image dimensions to process max 25 subtiles.
* Future versions will support automated tiling and resampling of raster layers.