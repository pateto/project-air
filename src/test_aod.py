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

	def test_swath2tif(self):
		name = 'MOD02HKM'
		object_name = 'MODIS_SWATH_Type_L1B'
		field_name = 'EV_500_RefSB'
		
		sara = SARA(input_folder, workspace_folder)
		
		self.assertTrue(sara.swath2tif(name, object_name, field_name))
	
	#def test_resample(self):		
	#	self.assertTrue(aod.resample(aod.mod09ga))
	
	#def test_convertHDF(self):		
	#	self.assertTrue(aod.convertHDF())
		
if __name__ == '__main__':
	unittest.main()