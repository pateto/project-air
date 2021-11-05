import unittest, aod, os, shutil
from os.path import join
from subprocess import call
from aod import SARA
import pdb

input_folder = 'C:\\AlvaroE\\air\\data'
workspace_folder = "C:\\AlvaroE\\air\\ws"

"""# Get files for test
files = os.listdir(aod.mod)
aod.geolocation = aod.getFile('MOD03', files)
aod.mod02hkm = aod.getFile('MOD02HKM', files)
aod.mod09ga = aod.getFile('MOD09GA', files)"""

class MyTest(unittest.TestCase):

	def setUp(self):
		if not os.path.exists(workspace_folder):
			os.makedirs(workspace_folder)
	
	def tearDown(self):		
		shutil.rmtree(workspace_folder)

	"""def test_resampling(self):
		sara = SARA(input_folder, workspace_folder)
		sara.reprojectFiles()"""
	
	def test_getRadianceScaleFactor(self):
		# Read GDAL Parameters
		sara = SARA(input_folder, workspace_folder)		
		print(sara.getRadianceScaleFactor())
		
if __name__ == '__main__':
	unittest.main()