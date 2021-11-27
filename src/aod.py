from osgeo import gdal, ogr, osr
import os, shutil, numpy, sys, math
from subprocess import call
from os.path import join
from scipy.optimize import fsolve
from scipy.ndimage import filters
import pdb

class SARA:

	def __init__(self, input_folder, workspace_folder):

		# init parameters
		self.input_folder = input_folder
		self.workspace_folder = workspace_folder		
		
		# get files
		files = os.listdir(input_folder)
		self.mod02hkm = self.getFile('MOD02HKM', files)
		self.mod03 = self.getFile('MOD03', files)
		self.mod35_l2 = self.getFile('MOD35_L2', files)
		self.mod09ga = self.getFile('MOD09GA', files)
		
		# geolocation file
		self.geolocation = self.mod03		
		
		# set tiff files
		self.mod02hkm_tiff = 'MOD02HKM_EV_500_RefSB.tif'
		self.mod03solarzenith_tiff = 'MOD03_SolarZenith.tif'
		self.mod03sensorzenith_tiff = 'MOD03_SensorZenith.tif'
		self.mod03solarazimuth_tiff = 'MOD03_SolarAzimuth.tif'
		self.mod03senzorazimuth_tiff = 'MOD03_SensorAzimuth.tif'
		self.mod03height_tiff = 'MOD03_Height.tif'
		self.mod35cloudmask_tiff = 'MOD35_Cloud_Mask.tif'
		self.mod09ga_tiff = 'MOD09GA.tif'
        
        # Enviromental variables		
		os.environ['MRTDATADIR'] = '/HEG/data'
		os.environ['PGSHOME'] = '/HEG/TOOLKIT_MTD'		
		os.environ['PATH'] = os.environ['PATH'] + ':/HEG/bin'
	
	def setBoundingBox(self, ulx, uly, lrx, lry):
		self.ulx = ulx
		self.uly = uly
		self.lrx = lrx
		self.lry = lry
	
	# get modis file
	def getFile(self, name, files):
		for f in files:
			if (f.startswith(name)):
				return f
	
	def degrees2decimal(self, d, m ,s):
		return d + m / 60.0 + s / 3600.0

	# reproject MOD09GA raster using heg
	def resample(self, name):
        
		if name == 'MOD09GA':
			input_file = join(self.input_folder, self.mod09ga)
		output_file = join(self.workspace_folder, name)
		
		# define parameters
		
		# TODO Check names
		input_p = 'INPUT_FILENAME = %s\n' % input_file
		output_p = 'OUTPUT_FILENAME = %s.tif\n' % output_file
		file = join(self.workspace_folder, 'resample.txt')

		# create parameter file
		
		fo = open(file, 'w', newline='')
		fo.write('NUM_RUNS = 1\n')
		fo.write('BEGIN\n')
		fo.write(input_p)
		fo.write('OBJECT_NAME = MODIS_Grid_500m_2D|\n')
		fo.write('FIELD_NAME = sur_refl_b04_1\n')
		fo.write('BAND_NUMBER = 1\n')
		fo.write('RESAMPLING_TYPE = NN\n')
		fo.write('OUTPUT_PROJECTION_TYPE = GEO\n')
		fo.write('ELLIPSOID_CODE = WGS84\n')
		fo.write(output_p)
		fo.write('OUTPUT_TYPE = GEO\n')
		fo.write('END\n')
		fo.close()
		
		# call resample (from heg)	
		call(['resample', '-p', file])
		
		return True

	# reproject raster using swath to tif
	def swath2tif(self, name, object_name, field_name):    
		
		if name == 'MOD02HKM':
			input_file = join(self.input_folder, self.mod02hkm)
		elif name == 'MOD03':
			input_file = join(self.input_folder, self.mod03)
		elif name == 'MOD35':
			input_file = join(self.input_folder, self.mod35_l2)
		
		# Write input parameters
		call(['hegtool', '-h', input_file])
		
		# Read input parameters
		header_parameters = {}
		header_file = open('/HEG/HegHdr.hdr')
		
		for line in header_file.readlines():
			if line!='\n':				
				dict_values = line.strip().split('=')
				header_parameters[dict_values[0]] = dict_values[1]
			
		header_file.close()
		
		
		# define parameters		
		input_p = 'INPUT_FILENAME = %s\n' % join(self.input_folder, input_file)
		output_p = 'OUTPUT_FILENAME = %s.tif\n' % join(self.workspace_folder, name + '_' + field_name)
		object_name_p = 'OBJECT_NAME = %s\n' % join(object_name)
		field_name_p = 'FIELD_NAME = %s|\n' % join(field_name)	
			
		# set geo parameters
		
		swath_lat_min = header_parameters['SWATH_LAT_MIN']
		swath_lat_max = header_parameters['SWATH_LAT_MAX']
		swath_lon_min = header_parameters['SWATH_LON_MIN']
		swath_lon_max = header_parameters['SWATH_LON_MAX']
		swath_x_pixel_res_degrees = header_parameters['SWATH_X_PIXEL_RES_DEGREES'] 
		swath_y_pixel_res_degrees = header_parameters['SWATH_Y_PIXEL_RES_DEGREES'] 
		
		
		ul_corner_p = 'SPATIAL_SUBSET_UL_CORNER = ( %s %s )\n' % (swath_lat_max, swath_lon_min)
		lr_corner_p = 'SPATIAL_SUBSET_LR_CORNER = ( %s %s )\n' % (swath_lat_min, swath_lon_max)
		
		pixel_size_x = 'OUTPUT_PIXEL_SIZE_X = %s\n' % (swath_x_pixel_res_degrees)
		pixel_size_y = 'OUTPUT_PIXEL_SIZE_Y = %s\n' % (swath_y_pixel_res_degrees)


		parameters_file = join(self.workspace_folder, 'swath2tif.txt')
		
		# create parameter file		
		fo = open(parameters_file, 'w', newline='')
		fo.write('NUM_RUNS = 1\n')
		fo.write('BEGIN\n')
		fo.write(input_p)
		fo.write(object_name_p)
		fo.write(field_name_p)
		fo.write('BAND_NUMBER = 1\n')
		fo.write(pixel_size_x)
		fo.write(pixel_size_y)
		fo.write(ul_corner_p)
		fo.write(lr_corner_p)
		fo.write('RESAMPLING_TYPE = NN\n')
		fo.write('OUTPUT_PROJECTION_TYPE = GEO\n')
		fo.write('ELLIPSOID_CODE = DEFAULT\n')
		fo.write('OUTPUT_PROJECTION_PARAMETERS = ( 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  )\n')
		fo.write(output_p)
		fo.write('OUTPUT_TYPE = GEO\n')
		fo.write('END\n')
		fo.close()
		
		# call swath2tif(from heg)	
		call(['swtif', '-p', parameters_file])
		
		return True
	

	# reproject files
	def reprojectFiles(self):

		# create workspace		
		if os.path.exists(self.workspace_folder):
			shutil.rmtree(self.workspace_folder)		
		os.makedirs(self.workspace_folder)
		
		# get SDS EV_500 (Band 2) from MOD02HKM
		self.swath2tif('MOD02HKM', 'MODIS_SWATH_Type_L1B', 'EV_500_RefSB')

		# get SDS SolarZenith from MOD03
		self.swath2tif('MOD03', 'MODIS_Swath_Type_GEO', 'SolarZenith')

		# get SDS SensorZenith from MOD03
		self.swath2tif('MOD03', 'MODIS_Swath_Type_GEO', 'SensorZenith')

		# get SDS SolarAzimuth from MOD03
		self.swath2tif('MOD03', 'MODIS_Swath_Type_GEO', 'SolarAzimuth')

		# get SDS SensorAzimuth from MOD03
		self.swath2tif('MOD03', 'MODIS_Swath_Type_GEO', 'SensorAzimuth')

		# get SDS Height from MOD03
		self.swath2tif('MOD03', 'MODIS_Swath_Type_GEO', 'Height')		

		# get the Cloud Mask (Byte 0) from MOD35
		self.swath2tif('MOD35', 'mod35', 'Cloud_Mask')	

		# Get the Surface reflectance from MOD09GA
		self.resample('MOD09GA')
		
		return True

	# clip raster
	def clipRaster(self):

		# define parameters

		input_file = join(self.workspace_folder, self.mod02hkm_tiff)
		output_file = join(self.workspace_folder, 'mask.tif')
		
		call(['gdal_translate', '-projwin', str(self.ulx), str(self.uly), str(self.lrx), str(self.lry), '-of', 'GTiff', input_file, output_file])
	
	# correct clipped raster
	def correctRaster(self):

		# Define data sources

		input_file = join(self.workspace_folder, 'mask.tif')
		output_file = join(self.workspace_folder, 'new_mask.tif')

		# Open raster

		raster = gdal.Open(input_file)

		# Get raster information

		number_x = raster.RasterXSize
		number_y = raster.RasterYSize
		data = raster.GetRasterBand(1).ReadAsArray(0, 0, number_x, number_y)

		# Create new mask

		format = "GTiff"
		driver = gdal.GetDriverByName(format)
		dst_raster = driver.Create(output_file, number_x, number_y, 1, gdal.GDT_Int16)
		dst_raster.SetGeoTransform(raster.GetGeoTransform())
		dst_raster.SetProjection(raster.GetProjectionRef())

		# Write data

		dst_raster.GetRasterBand(1).WriteArray( data )

		# Close raster

		dst_raster = None

	# raster to points
	def rasterToPoints(self):

		# Define data sources

		input_file = join(self.workspace_folder, 'new_mask.tif')
		output_file = join(self.workspace_folder, 'points.shp')

		# Setup raster

		raster = gdal.Open(input_file)

		# Get raster information

		number_x = raster.RasterXSize
		number_y = raster.RasterYSize
		band = raster.GetRasterBand(1)
		(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = raster.GetGeoTransform()
		data = band.ReadAsArray(0, 0, number_x, number_y)

		# Setup shapefile

		driver = ogr.GetDriverByName('ESRI Shapefile')
		shp_file = driver.CreateDataSource(output_file)

		# Define spatial reference

		srs = osr.SpatialReference()
		srs.ImportFromWkt(raster.GetProjection())

		# Define layer

		layer = shp_file.CreateLayer('AOD', srs, ogr.wkbPoint)

		# Define attributes

		field_def = ogr.FieldDefn('gid', ogr.OFTInteger)
		layer.CreateField(field_def)
		
		field_def = ogr.FieldDefn('refsb_b1', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('height', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('snrazimuth', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('snrzenith', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('solazimuth', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('solzenith', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('surrefl_b4', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('cloud_b0', ogr.OFTInteger)
		layer.CreateField(field_def)

		field_def = ogr.FieldDefn('aod', ogr.OFTReal)
		layer.CreateField(field_def)

		# Iterate over columns and rows

		gid = 0
		
		for i in range(0, number_x):
			for j in range(0, number_y):
			
				# Get middle of the pixel (adding half the cell size to center the point)
				
				x = i * x_size + upper_left_x + (x_size / 2)
				y = j * y_size + upper_left_y + (y_size / 2)
				
				# Create feature
				
				feature = ogr.Feature(layer.GetLayerDefn())		
				wkt = 'POINT(%f %f)' % (x, y)
				point = ogr.CreateGeometryFromWkt(wkt)
				feature.SetField('gid', gid)
				feature.SetGeometry(point)
				layer.CreateFeature(feature)
				feature.Destroy()
				
				gid += 1
				
		shp_file.Destroy()
	
	# set point values from raster	
	def setPointsValues(self, shp_field, raster_name):	

		# Define data sources

		raster_name = join(self.workspace_folder, raster_name)
		shp_name = join(self.workspace_folder, 'points.shp')

		#open points layer

		driver = ogr.GetDriverByName('ESRI Shapefile')
		shp_file = driver.Open(shp_name, True)
		layer = shp_file.GetLayer(0)

		#open raster layer

		raster = gdal.Open(raster_name) 
		
		# Define spatial reference
		srs = osr.SpatialReference()
		srs.SetWellKnownGeogCS('WGS84')
		raster.SetProjection(srs.ExportToWkt())
		
		# Get information
		
		nx = raster.RasterXSize
		ny = raster.RasterYSize
		(upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = raster.GetGeoTransform()
		band = raster.GetRasterBand(1)
		data = band.ReadAsArray(0, 0, nx, ny)

		for feature in layer:

			# Get point centroid
			
			geom = feature.GetGeometryRef()
			feature_x = geom.Centroid().GetX()
			feature_y = geom.Centroid().GetY()
			
			# Get pixel value
			
			i = int((feature_x - upper_left_x)/ x_size)
			j = int((feature_y - upper_left_y)/ y_size)
			
			# Set data value
			
			value = data[j, i]
			feature.SetField(shp_field, int(value))
			layer.SetFeature(feature)
			
		raster = None
		shp_file = None
	
	# update points shapefile
	def updatePoints(self):
	
		points_dict = {'refsb_b1' : self.mod02hkm_tiff,
				'height' : self.mod03height_tiff,
				'snrazimuth' : self.mod03senzorazimuth_tiff,
				'snrzenith' : self.mod03sensorzenith_tiff,
				'solazimuth' : self.mod03solarazimuth_tiff,
				'solzenith' : self.mod03solarzenith_tiff,
				'surrefl_b4' : self.mod09ga_tiff,
				'cloud_b0' : self.mod35cloudmask_tiff}

		# Loop over the dictionary and set the point values in the shapefile
		for elem in points_dict:
			print(elem, points_dict[elem])
			self.setPointsValues(elem, points_dict[elem])

	# is cloud free?
	def isCloudFree(self, cloudMask):

		# Convert to binary string
		b = bin(cloudMask & 0b11111111)

		# compare last 3 digits
		# It is confident clear?
		if(b[7:] == "111"):
			ans = True
		else:
			ans = False

		return ans

	# make AOD calculation
	def makeCalculation(self, values):

		# Reading values

		# TOA Radiance
		TOARadiance = float(values['refsb_b1']) * self.TOARadianceScaleFactor

		# Solar Zenith Angle
		solarZenithAngle = float(values['solzenith']) * 0.01

		# Sensor Zenith Angle
		sensorZenithAngle = float(values['snrzenith']) * 0.01

		# Solar Azimuth Angle
		solarAzimuthAngle = float(values['solazimuth']) * 0.01

		# Sensor Azimuth Angle
		sensorAzimuthAngle = float(values['snrazimuth']) * 0.01

		# Height
		height = float(values['height']) / 1000

		# Surface Reflectance
		surfaceReflectance = float(values['surrefl_b4']) * 0.0001

		# Calculate Top of Atmosphere (TOA) Reflectance
		TOAReflectance = (math.pi * TOARadiance * math.pow(self.distance, 2))/(self.ESUN*math.cos(math.radians(solarZenithAngle)))

		# Calculate cosine solar and sensor zenith angles
		cosSolarZenithAngle = math.cos(math.radians(solarZenithAngle))
		cosSensorZenithAngle = math.cos(math.radians(sensorZenithAngle))

		# Calculate sine solar and sensor zenith angles
		sinSolarZenithAngle = math.sin(math.radians(solarZenithAngle))
		sinSensorZenithAngle = math.sin(math.radians(sensorZenithAngle))

		# Calculate cosine relative Azimuth angles
		cosRelativeAzimuthAngles = math.cos(math.radians(solarAzimuthAngle-sensorAzimuthAngle))

		# Calculate Scattering Phase Angle		
		scatteringPhaseAngle = math.degrees(math.acos(cosSolarZenithAngle * cosSensorZenithAngle + sinSolarZenithAngle * sinSensorZenithAngle * cosRelativeAzimuthAngles))

		# Calculate ambient pressure with respect to elevation
		ambientPressure = 1013.25 * math.exp(-0.000118558 * height * 1000)

		# Calculate Rayleigh Optical Depth
		rayleighOpticalDepth = (ambientPressure/1013.25)*(0.00864 + height*0.0000065)*11.243728

		# Calculate Rayleigh phase function
		rayleighPhaseFunction = 0.06050422+0.0572197*pow(math.cos(math.radians(scatteringPhaseAngle)),2)

		# Calculate Rayleigh reflectance (from multiple scattering in the absence of aerosols)
		rayleighReflectance = (math.pi * rayleighOpticalDepth * rayleighPhaseFunction) / (cosSolarZenithAngle * cosSensorZenithAngle)

		# Calculate Aerosol scattering phase function
		asymmetricParameter2 = math.pow(self.asymmetricParameter,2)
		aerosolScatteringPhaseFunction = (1-asymmetricParameter2)/math.pow(1+asymmetricParameter2-2*self.asymmetricParameter*math.cos(math.pi-math.radians(scatteringPhaseAngle)),1.5)

		# Calculate Aerosol Optical Depth Coefficient
		AODCoef = cosSolarZenithAngle*cosSensorZenithAngle/(self.singleScatteringAlbedo*aerosolScatteringPhaseFunction)

		# Calculate Aerosol Optical Depth!
		def setAOD(aerosolOpticalDepth):
			return AODCoef*(TOAReflectance-rayleighReflectance-(math.exp(-(rayleighOpticalDepth+aerosolOpticalDepth)/cosSolarZenithAngle)*math.exp(-(rayleighOpticalDepth+aerosolOpticalDepth)/cosSensorZenithAngle)*surfaceReflectance / (1 - surfaceReflectance * ((0.92*rayleighOpticalDepth+(1-self.asymmetricParameter)*aerosolOpticalDepth)*math.exp(-(rayleighOpticalDepth+aerosolOpticalDepth)))))) - aerosolOpticalDepth

		aerosolOpticalDepth = fsolve(setAOD,1)
		
		return aerosolOpticalDepth[0]		

	# get the distance from the sun to the earth
	def getEarthSunDistance(self, day):
		earthSunDistance = {'001':0.98331,'002':0.9833,'003':0.9833,'004':0.9833,'005':0.9833,'006':0.98332,'007':0.98333,'008':0.98335,'009':0.98338,'010':0.98341,'011':0.98345,'012':0.98349,'013':0.98354,'014':0.98359,'015':0.98365,'016':0.98371,'017':0.98378,'018':0.98385,'019':0.98393,'020':0.98401,'021':0.9841,'022':0.98419,'023':0.98428,'024':0.98439,'025':0.98449,'026':0.9846,'027':0.98472,'028':0.98484,'029':0.98496,'030':0.98509,'031':0.98523,'032':0.98536,'033':0.98551,'034':0.98565,'035':0.9858,'036':0.98596,'037':0.98612,'038':0.98628,'039':0.98645,'040':0.98662,'041':0.9868,'042':0.98698,'043':0.98717,'044':0.98735,'045':0.98755,'046':0.98774,'047':0.98794,'048':0.98814,'049':0.98835,'050':0.98856,'051':0.98877,'052':0.98899,'053':0.98921,'054':0.98944,'055':0.98966,'056':0.98989,'057':0.99012,'058':0.99036,'059':0.9906,'060':0.99084,'061':0.99108,'062':0.99133,'063':0.99158,'064':0.99183,'065':0.99208,'066':0.99234,'067':0.9926,'068':0.99286,'069':0.99312,'070':0.99339,'071':0.99365,'072':0.99392,'073':0.99419,'074':0.99446,'075':0.99474,'076':0.99501,'077':0.99529,'078':0.99556,'079':0.99584,'080':0.99612,'081':0.9964,'082':0.99669,'083':0.99697,'084':0.99725,'085':0.99754,'086':0.99782,'087':0.99811,'088':0.9984,'089':0.99868,'090':0.99897,'091':0.99926,'092':0.99954,'093':0.99983,'094':1.00012,'095':1.00041,'096':1.00069,'097':1.00098,'098':1.00127,'099':1.00155,'100':1.00184,'101':1.00212,'102':1.0024,'103':1.00269,'104':1.00297,'105':1.00325,'106':1.00353,'107':1.00381,'108':1.00409,'109':1.00437,'110':1.00464,'111':1.00492,'112':1.00519,'113':1.00546,'114':1.00573,'115':1.006,'116':1.00626,'117':1.00653,'118':1.00679,'119':1.00705,'120':1.00731,'121':1.00756,'122':1.00781,'123':1.00806,'124':1.00831,'125':1.00856,'126':1.0088,'127':1.00904,'128':1.00928,'129':1.00952,'130':1.00975,'131':1.00998,'132':1.0102,'133':1.01043,'134':1.01065,'135':1.01087,'136':1.01108,'137':1.01129,'138':1.0115,'139':1.0117,'140':1.01191,'141':1.0121,'142':1.0123,'143':1.01249,'144':1.01267,'145':1.01286,'146':1.01304,'147':1.01321,'148':1.01338,'149':1.01355,'150':1.01371,'151':1.01387,'152':1.01403,'153':1.01418,'154':1.01433,'155':1.01447,'156':1.01461,'157':1.01475,'158':1.01488,'159':1.015,'160':1.01513,'161':1.01524,'162':1.01536,'163':1.01547,'164':1.01557,'165':1.01567,'166':1.01577,'167':1.01586,'168':1.01595,'169':1.01603,'170':1.0161,'171':1.01618,'172':1.01625,'173':1.01631,'174':1.01637,'175':1.01642,'176':1.01647,'177':1.01652,'178':1.01656,'179':1.01659,'180':1.01662,'181':1.01665,'182':1.01667,'183':1.01668,'184':1.0167,'185':1.0167,'186':1.0167,'187':1.0167,'188':1.01669,'189':1.01668,'190':1.01666,'191':1.01664,'192':1.01661,'193':1.01658,'194':1.01655,'195':1.0165,'196':1.01646,'197':1.01641,'198':1.01635,'199':1.01629,'200':1.01623,'201':1.01616,'202':1.01609,'203':1.01601,'204':1.01592,'205':1.01584,'206':1.01575,'207':1.01565,'208':1.01555,'209':1.01544,'210':1.01533,'211':1.01522,'212':1.0151,'213':1.01497,'214':1.01485,'215':1.01471,'216':1.01458,'217':1.01444,'218':1.01429,'219':1.01414,'220':1.01399,'221':1.01383,'222':1.01367,'223':1.01351,'224':1.01334,'225':1.01317,'226':1.01299,'227':1.01281,'228':1.01263,'229':1.01244,'230':1.01225,'231':1.01205,'232':1.01186,'233':1.01165,'234':1.01145,'235':1.01124,'236':1.01103,'237':1.01081,'238':1.0106,'239':1.01037,'240':1.01015,'241':1.00992,'242':1.00969,'243':1.00946,'244':1.00922,'245':1.00898,'246':1.00874,'247':1.0085,'248':1.00825,'249':1.008,'250':1.00775,'251':1.0075,'252':1.00724,'253':1.00698,'254':1.00672,'255':1.00646,'256':1.0062,'257':1.00593,'258':1.00566,'259':1.00539,'260':1.00512,'261':1.00485,'262':1.00457,'263':1.0043,'264':1.00402,'265':1.00374,'266':1.00346,'267':1.00318,'268':1.0029,'269':1.00262,'270':1.00234,'271':1.00205,'272':1.00177,'273':1.00148,'274':1.00119,'275':1.00091,'276':1.00062,'277':1.00033,'278':1.00005,'279':0.99976,'280':0.99947,'281':0.99918,'282':0.9989,'283':0.99861,'284':0.99832,'285':0.99804,'286':0.99775,'287':0.99747,'288':0.99718,'289':0.9969,'290':0.99662,'291':0.99634,'292':0.99605,'293':0.99577,'294':0.9955,'295':0.99522,'296':0.99494,'297':0.99467,'298':0.9944,'299':0.99412,'300':0.99385,'301':0.99359,'302':0.99332,'303':0.99306,'304':0.99279,'305':0.99253,'306':0.99228,'307':0.99202,'308':0.99177,'309':0.99152,'310':0.99127,'311':0.99102,'312':0.99078,'313':0.99054,'314':0.9903,'315':0.99007,'316':0.98983,'317':0.98961,'318':0.98938,'319':0.98916,'320':0.98894,'321':0.98872,'322':0.98851,'323':0.9883,'324':0.98809,'325':0.98789,'326':0.98769,'327':0.9875,'328':0.98731,'329':0.98712,'330':0.98694,'331':0.98676,'332':0.98658,'333':0.98641,'334':0.98624,'335':0.98608,'336':0.98592,'337':0.98577,'338':0.98562,'339':0.98547,'340':0.98533,'341':0.98519,'342':0.98506,'343':0.98493,'344':0.98481,'345':0.98469,'346':0.98457,'347':0.98446,'348':0.98436,'349':0.98426,'350':0.98416,'351':0.98407,'352':0.98399,'353':0.98391,'354':0.98383,'355':0.98376,'356':0.9837,'357':0.98363,'358':0.98358,'359':0.98353,'360':0.98348,'361':0.98344,'362':0.9834,'363':0.98337,'364':0.98335,'365':0.98333,'366':0.98331}
		return earthSunDistance[day]

	# get radiance scale factor
	def getRadianceScaleFactor(self):

		# converted hdf4 to hdf5
		file = join(self.input_folder, self.mod02hkm)
		
		subset_file = 'HDF4_EOS:EOS_SWATH:"%s":MODIS_SWATH_Type_L1B:EV_500_RefSB' % file
		
		gdal_file = gdal.Open(subset_file)
		
		radiance_scales = gdal_file.GetMetadataItem('radiance_scales')
		
		radiance_scale_factor = float(radiance_scales.split(',')[0])
		
		return radiance_scale_factor
		

	# calculate AOD
	def calculateAOD(self):
		
		# single scattering albedo
		self.singleScatteringAlbedo = 0.8170
		
		# asymmetric parameter
		self.asymmetricParameter = 0.6889

		# MOD02HKM ref 500m scale factor
		self.TOARadianceScaleFactor = self.getRadianceScaleFactor()

		# Earth sun distance		
		self.distance = self.getEarthSunDistance(self.mod02hkm[14:17])

		# mean solar exoatmospheric radiance
		self.ESUN = 1850

		#open points layer
		
		srcFile = join(self.workspace_folder, 'points.shp')
		driver = ogr.GetDriverByName('ESRI Shapefile')
		shpFile = driver.Open(srcFile, True)
		layer = shpFile.GetLayer(0)
		
		total = float(layer.GetFeatureCount())
		
		for feature in layer:
		
			# get values
			values = {}
			values['refsb_b1'] = feature.GetField('refsb_b1')
			values['height'] = feature.GetField('height')
			values['snrazimuth'] = feature.GetField('snrazimuth')			
			values['snrzenith'] = feature.GetField('snrzenith')
			values['solazimuth'] = feature.GetField('solazimuth')
			values['solzenith'] = feature.GetField('solzenith')
			values['surrefl_b4'] = feature.GetField('surrefl_b4')
			values['cloud_b0'] = feature.GetField('cloud_b0')
			
			# Get cloud mask
			cloudMask = int(values['cloud_b0'])

			# Is it cloud free?
			if self.isCloudFree(cloudMask):				
				feature.SetField('aod', self.makeCalculation(values))				
			else:            
				feature.SetField('aod', -9999)
				
			layer.SetFeature(feature)
				
			# print advance			
			i = feature.GetField('gid')
			per = int(i / total * 100)			
			sys.stdout.write("\r%d%%" % per)
			sys.stdout.flush()
			
		
		sys.stdout.write("\r%d%%\n" % 100)
		sys.stdout.flush()		
		shpFile.Destroy()
	
	# create raster
	def createRaster(self):

		# Open the shapefile
		
		srcFile = join(self.workspace_folder, 'points.shp')
		driver = ogr.GetDriverByName('ESRI Shapefile')
		shpFile = driver.Open(srcFile, True)
		layer = shpFile.GetLayer(0)
		
		# Open mask file
		
		input_file = join(self.workspace_folder, 'new_mask.tif')		
		raster = gdal.Open(input_file, True)

		# Get raster information

		number_x = raster.RasterXSize
		number_y = raster.RasterYSize
		band = raster.GetRasterBand(1)		
		data = band.ReadAsArray(0, 0, number_x, number_y)
		
		# create new matrix
		k = 0		
		new_data = data.astype(float)
		for i in range(0, number_x):
			for j in range(0, number_y):
				new_data[j,i] = layer[k].GetField('aod')
				k += 1
		
		shpFile = None
		
		# create new raster
		
		output_file = join(self.workspace_folder, 'aod.tif')
		format = "GTiff"
		driver = gdal.GetDriverByName(format)
		dst_raster = driver.Create(output_file, number_x, number_y, 1, gdal.GDT_Float64)

		# Redefine the origin coordinates
		
		dst_raster.SetGeoTransform(raster.GetGeoTransform())

		# Set spatial reference
		
		dst_raster.SetProjection(raster.GetProjectionRef())

		# Write data

		dst_raster.GetRasterBand(1).WriteArray( new_data )

		# Close raster

		raster = None
		dst_raster = None
	
	# create raster
	def medianFilter(self):

		# Open aod file

		input_file = join(self.workspace_folder, 'aod.tif')
		raster = gdal.Open(input_file, True)

		# Get raster information

		number_x = raster.RasterXSize
		number_y = raster.RasterYSize
		band = raster.GetRasterBand(1)		
		data = band.ReadAsArray(0, 0, number_x, number_y)

		# apply filter
		
		new_data = filters.median_filter(data, 9)

		# create new raster

		output_file = join(self.workspace_folder, 'aod_m9.tif')
		format = "GTiff"
		driver = gdal.GetDriverByName(format)
		dst_raster = driver.Create(output_file, number_x, number_y, 1, gdal.GDT_Float64)
		dst_raster.SetGeoTransform(raster.GetGeoTransform())
		dst_raster.SetProjection(raster.GetProjectionRef())
		dst_raster.GetRasterBand(1).WriteArray(new_data)

		# Close raster

		raster = None
		dst_raster = None


	def interpolate(self, m, x, b):
		return m*x + b

	def base(self, x):
		if x <= .125:
			return 0
		elif x <= 0.375:
			return self.interpolate(4, x, -.5)
		elif x <= .625:
			return 1.0
		elif x <= .875:
			return self.interpolate(-4, x, 3.5)
		else:
			return 0


	def red(self, gray):
		return self.base(gray - .25)

	def green(self, gray):
		return self.base(gray)

	def blue(self, gray):
		return self.base(gray + .25)
	
	# create RGB raster
	def rgbRaster(self):
	
		# Open aod file
		input_file = join(self.workspace_folder, 'aod_m9.tif')
		raster = gdal.Open(input_file, True)

		# Get raster information

		number_x = raster.RasterXSize
		number_y = raster.RasterYSize
		band = raster.GetRasterBand(1)		
		data = band.ReadAsArray(0, 0, number_x, number_y)

		# create RGB bands
		red_band = numpy.zeros((number_y, number_x), dtype=numpy.uint8)
		green_band = numpy.zeros((number_y, number_x), dtype=numpy.uint8)
		blue_band = numpy.zeros((number_y, number_x), dtype=numpy.uint8)
		
		max = data.max()
		
		for i in range(0, number_x):
			for j in range(0, number_y):
			
				if data[j, i]==-9999.0:
					red_band[j, i] = 255
					green_band[j, i] = 255
					blue_band[j, i] = 255
				else:
					gray = data[j, i] / max
					red_band[j, i] = self.red(gray) * 255
					green_band[j, i] = self.green(gray) * 255
					blue_band[j, i] = self.blue(gray) * 255

		# create new raster

		output = join(self.workspace_folder, 'aod_m9_rgb.tif')
		format = "GTiff"
		driver = gdal.GetDriverByName(format)
		dst_raster = driver.Create(output, number_x, number_y, 3, gdal.GDT_Byte)
		dst_raster.SetGeoTransform(raster.GetGeoTransform())
		dst_raster.SetProjection(raster.GetProjectionRef())
		dst_raster.GetRasterBand(1).WriteArray(red_band)
		dst_raster.GetRasterBand(2).WriteArray(green_band)
		dst_raster.GetRasterBand(3).WriteArray(blue_band)
		dst_raster.FlushCache()

		# Close raster

		raster = None
		dst_raster = None