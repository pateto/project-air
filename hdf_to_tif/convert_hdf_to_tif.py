import pdb
import os

# This script converts hdf files to tif files.
# It is designed for windows
# The software HEG must be installed

def init():

    # Set enviroment variables
    os.environ["CYGWIN"] = "nodosfilewarning"
    os.environ["LD_LIBRARY_PATH"] = "c:\\heg\\HEG_Win\\bin"
    os.environ["MRTDATADIR"] = "c:\\heg\\HEG_Win\\data"
    os.environ["MRTBINDIR"] = "c:\\heg\\HEG_Win\\bin"
    os.environ["PGSHOME"] = "c:\\heg\\HEG_Win\\TOOLKIT_MTD"
    os.environ["HEGUSER"] = "heg"

    read_files()

# Lee el nombre de cada archivo
def read_files():
    file = "list.txt"
    lines = open(file)
    for filename in lines:
    
        print(filename)
        
        # Create params file
        create_params(filename.strip())
        
        # Run heg
        run_swtif()
        
    
# crear un archivo de parametros
def create_params(input_filename):

    output_filename = input_filename.replace("AEROSOL_3K", "AEROSOL_3K_TIF")
    output_filename = output_filename.replace(".hdf", ".tif")
    
    dir_output = os.path.dirname(output_filename)
    
    # create output directory if it doesn't exists
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    
    params_txt = """
    
NUM_RUNS = 1

BEGIN
INPUT_FILENAME = {0}
OBJECT_NAME = mod04
FIELD_NAME = Corrected_Optical_Depth_Land|
BAND_NUMBER = 2
OUTPUT_PIXEL_SIZE_X = 0.027439
OUTPUT_PIXEL_SIZE_Y = 0.027084
SPATIAL_SUBSET_UL_CORNER = ( 5.372773 -77.111309 )
SPATIAL_SUBSET_LR_CORNER = ( 2.655314 -74.899425 )
RESAMPLING_TYPE = NN
OUTPUT_PROJECTION_TYPE = GEO
ELLIPSOID_CODE = DEFAULT
OUTPUT_PROJECTION_PARAMETERS = ( 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  )
OUTPUT_FILENAME = {1}
OUTPUT_TYPE = GEO
END

""".format(input_filename, output_filename)
    #print(params_txt)
    file = open("params_swath", 'w', newline='\n')
    file.write(params_txt)
    file.close()

# ejecutar swtif
def run_swtif():
    # execute command
    cmd = "C:\\heg\\HEG_Win\\bin\\swtif.exe -P C:\\AEROSOL_3K\\script\\params_swath"
    os.system(cmd)
    
# main
init()
