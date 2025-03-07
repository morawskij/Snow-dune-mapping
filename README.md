# Programmatic tools for automated mapping of snow dunes based on a DEM

## Additionnaly:

## Raster-skeletonization
Python tools for thin grayscale raster skeletonization, created because the QGIS plugin is bugged. Algorithm implemented as described in Biagioni & Eriksson 2012 paper *Map Inference in the Face of Noise and Disparity*

## Important remarks:
- The *nodata_value* stored in the **raster_data** class object should be a number lower than the minimum value encountered within that raster (also after any arithmetic operations you might wanna apply, e. g. if the minimum of the raster is 0 and maximum of the raster is 100 but you intend to multiply it by -1 at some point, be sure to set *nodata_value* to a number below -100)
- Functions *raster_data.add* and *raster_data.multiply_by* allow basic arithmetic operations on and between rasters. In particular, since skeletonization always runs in a top-down fashion, be sure to multiply the raster by -1 (and optionally shift) if interested in skeletonizing starting from the minima
- With the arithmetic functions, it is the nodata pixels (and nodata value) of the object you call the action on, not the one you put as argument, that are being kept as nodata, keep it in mind when applying binary masks etc.  
- Parameters *crop_x* and *crop_y* can serve either to crop a raster (*raster_data.crop*) or just to constrain the visualization to a smaller area, while processing the entire raster (*raster_data.skeletonization*, (*raster_data.visualize*))
- If the *boolean* argument of the *raster_data.skeletonization* function is set to *True*, all non-zero values of the skeleton are overwritten by zero, in other words, this is relevant to the cases when we are interested in the overall shape of the skeleton and not the differentiation between the levels. A better practice is to keep *boolean=False* and then use tresholding before exporting to set all non-zero pixels of a skeleton to 1. Like this, we keep all the information and we can for example export a partial skeleton by setting a higher treshold without a need for recomputing the skeleton
- The *verbose_concerned* argument of the *raster_data.skeletonization* function, set to *False* by default, when set to *True* causes the algorithm to print updates upon entering each iteration - this is to check that the program is not completely stuck when processing a large raster
- The *verbose* argument of the *raster_data.skeletonization* set to *True* by default, causes the algorithm to inform about major progress (entering another grayscale level of skeletonization)
- The *visual_control* argument of the *raster_data.skeletonization* set to *True* by default, causes the algorithm to visually represent a starting raster of each skeletonization phase (using *cmap2*), next to the original raster (using *cmap1*) and the parital skeleton obtained upon completion of that step (*cmap3*)
- Function *raster_data.write_single_band_tif* takes a parameter *dtype*, set to *None* by default. It will then automatically check whether the raster is of float or int type, and set the data type accordingly. This is done based on the maximum of the raster and might therefore lead to undesirable results if the maximum of a raster which contains non-integer values is coincidentially an integer value. In that case, set *dtype=rasterio.float32* manually
