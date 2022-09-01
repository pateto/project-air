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

# Aerosol types
# Continental Clean, Continental Average, Continental Polluted, Urban
models = ['CCL', 'CAV', 'CPO', 'URB']
# models = ['URB']

for model in models:

    print( '---> set AOD!')
    sara.calculateAOD(model)

    print('---> create raster')
    sara.createRaster()

    for size in range(3,10,2): # window size for the medial filter: 3, 5, 7, 9

        print('---> apply median filter')
        sara.medianFilter(size)

        # print('---> create RGB image')
        # sara.rgbRaster(size)

print('---> end!')