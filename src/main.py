from aod import SARA
import pdb

#-----------------------------------------#
#                   main
#-----------------------------------------#

input_folder = '/project-air/input'
workspace_folder = '/project-air/output'

#ulx = -76.33793
#uly = 3.57850
#lrx = -76.25314
#lry = 3.49749

ulx = -74.5320
uly = 5.0293
lrx = -73.7022
lry = 4.1171

# Aerosol type (Continental clean, Continental average, Continental polluted, Urban)
aerosol_type = 'Urban'

"""
ulx = -sara.degrees2decimal(77, 31, 52)
uly = sara.degrees2decimal(8, 59, 54)
lrx = -sara.degrees2decimal(71, 49, 15)
lry = sara.degrees2decimal(3, 29, 44)
"""

sara = SARA(input_folder, workspace_folder, aerosol_type)
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