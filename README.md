# project-air

## Productos MODIS Resumen
https://lance.modaps.eosdis.nasa.gov/modis/

## Cobertura de nubes
http://worldview.earthdata.nasa.gov

## Descarga de productos
http://search.earthdata.nasa.gov

## HEG Reprojection tool
https://hdfeos.org/software/heg.php

windows: hegWINv2.15.Build9.8.zip
linux: hegLNX64v2.15.Build9.8.tar.gz

# docker installation

docker-compose up -d

## useful commands

docker build . -t <image_name>

docker run -d --name <container_name> <image_name>

docker exec -it <container_name> bash


# Files example
MOD02HKM.A2021110.1515.061.2021111022722.hdf
MOD03.A2021110.1515.061.2021110230802.hdf
MOD35_L2.A2021110.1515.061.2021111023002.hdf
MOD09GA.A2021110.h10v08.061.2021112034039.hdf