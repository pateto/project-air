import unittest, aod, os, shutil
from os.path import join
from subprocess import call
from aod import SARA
import pdb

input_folder = '/project-air/input'
workspace_folder = '/project-air/output'

class MyTest(unittest.TestCase):

	def setUp(self):
		if not os.path.exists(workspace_folder):
			os.makedirs(workspace_folder)			

	def test_resampling_MOD02HKM(self):
		sara = SARA(input_folder, workspace_folder)
		sara.swath2tif('MOD02HKM', 'MODIS_SWATH_Type_L1B', 'EV_500_RefSB')
        
	def test_resampling_MOD09GA(self):
		sara = SARA(input_folder, workspace_folder)
		sara.resample('MOD09GA')
	
	def test_getRadianceScaleFactor(self):
		# Read GDAL Parameters
		sara = SARA(input_folder, workspace_folder)		
		print(sara.getRadianceScaleFactor())
		
if __name__ == '__main__':
	unittest.main()