import aod
import pudb

#-----------------------------------------#
#                   main
#-----------------------------------------#

aod.mod = '/home/ws/air/mod/A2017093.1550/'
aod.img = '/home/ws/air/img/A2017093.1550/'
aod.ulx = -74.413
aod.uly = 5.038
aod.lrx = -73.791
aod.lry = 4.314
#aod.ulx = -aod.degrees2decimal(77, 31, 52)
#aod.uly = aod.degrees2decimal(8, 59, 54)
#aod.lrx = -aod.degrees2decimal(71, 49, 15)
#aod.lry = aod.degrees2decimal(3, 29, 44)

#pudb.set_trace()

print '---> init parameters'
aod.init()

print '---> create reprojection tiff files'
aod.reprojectFiles()

print '---> clip raster'
aod.clipRaster()

print '---> correct raster'
aod.correctRaster()

print '---> raster to points'
aod.rasterToPoints()

print '---> update points'
aod.updatePoints()

print '---> set AOD!'
aod.calculateAOD()

print '---> create raster'
aod.createRaster()

print '---> apply median filter'
aod.medianFilter()

print '---> create RGB image'
aod.rgbRaster()

print '---> end!'