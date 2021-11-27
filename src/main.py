from aod import SARA
import pdb

#-----------------------------------------#
#                   main
#-----------------------------------------#

input_folder = '/project-air/input'
workspace_folder = '/project-air/output'

ulx = -74.413
uly = 5.038
lrx = -73.791
lry = 4.314

"""
ulx = -sara.degrees2decimal(77, 31, 52)
uly = sara.degrees2decimal(8, 59, 54)
lrx = -sara.degrees2decimal(71, 49, 15)
lry = sara.degrees2decimal(3, 29, 44)
"""

sara = SARA(input_folder, workspace_folder)
sara.setBoundingBox(ulx, uly, lrx, lry)

print('---> create reprojection tiff files')
sara.reprojectFiles()

print('---> clip raster')
sara.clipRaster()

print('---> correct raster')
sara.correctRaster()

print('---> raster to points')
sara.rasterToPoints()

print('---> update points')
sara.updatePoints()

print( '---> set AOD!')
sara.calculateAOD()

print('---> create raster')
sara.createRaster()

print('---> apply median filter')
sara.medianFilter()

print('---> create RGB image')
sara.rgbRaster()

print('---> end!')