import unittest, aod, os, shutil
from os.path import join
from subprocess import call
from pudb import set_trace
#set_trace()

aod.ws = "/home/ws/air/test"
aod.mod = '/home/ws/air/mod/A2016033.1515/'

# Get files for test
files = os.listdir(aod.mod)
aod.geolocation = aod.getFile('MOD03', files)
aod.mod02hkm = aod.getFile('MOD02HKM', files)
name = 'MOD02HKM'
sds = 'EV_500_RefSB, 0, 1'
aod.mod09ga = aod.getFile('MOD09GA', files)

class MyTest(unittest.TestCase):

	def setUp(self):
		if not os.path.exists(aod.ws):
			os.makedirs(aod.ws)
	
	def tearDown(self):		
		shutil.rmtree(aod.ws)

	def test_swath2grid(self):		
		self.assertTrue(aod.swathReprojection(aod.mod02hkm, name, sds))
	
	def test_hegReprojection(self):
		self.assertTrue(aod.hegReprojection('MOD09GA'))
	
	def test_convertHDF(self):		
		self.assertTrue(aod.convertHDF())
		
if __name__ == '__main__':
	unittest.main()