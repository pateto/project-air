FROM osgeo/gdal
COPY hegLNX64v2.15.Build9.8.tar.gz /

RUN apt-get update
RUN apt-get install libgomp1
RUN apt-get install -y python3-pip
RUN pip install scipy

RUN tar -xf hegLNX64v2.15.Build9.8.tar.gz
RUN mkdir HEG
RUN mv heg.tar /HEG
RUN tar -xf /HEG/heg.tar -C /HEG
 
ENTRYPOINT ["tail", "-f", "/dev/null"]