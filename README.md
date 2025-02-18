# Raster-skeletonization
Standalone Python tool for thin grayscale raster skeletonization, created because the QGIS plugin is bugged. Algorithm implemented as described in Biagioni & Eriksson 2012 paper *Map Inference in the Face of Noise and Disparity*

## Important remarks:
- The *nodata_value* stored in the **raster_data** class object should be a number lower than the minimum value encountered within that raster
- Functions *raster_data.add* and *raster_data.multiply_by* allow basic arithmetic operations on and between rasters. In particular, since skeletonization always runs in a top-down fashion, be sure to multiply the raster by -1 (and optionally shift) if interested in skeletonizing starting from the minima   
- Parameters *crop_x* and *crop_y* can serve either to crop a raster (*raster_data.crop*) or just to constrain the visualization to a smaller area, while processing the entire raster (*raster_data.skeletonization*, (*raster_data.visualize*))
- If the *boolean* argument of the *raster_data.skeletonization* function is set to *True*, all non-zero values of the skeleton are overwritten by zero, in other words, this is relevant to the cases when we are interested in the overall shape of the skeleton and not the differentiation between the levels
- The *verbose_concerned* argument of the *raster_data.skeletonization* function, set to *False* by default, when set to *True* causes the algorithm to print updates upon entering each iteration - this is to check that the program is not completely stuck when processing a large raster
- The *verbose* argument of the *raster_data.skeletonization* set to *True* by default, causes the algorithm to inform about major progress (entering another grayscale level of skeletonization)
- The *visual_control* argument of the *raster_data.skeletonization* set to *False* by default, causes the algorithm to visually represent a starting raster of each skeletonization phase (using *cmap2*), next to the original raster for comparizon (using *cmap1*)
