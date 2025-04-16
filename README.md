# insarmaps_scripts

Collection of scripts to upload and manipulate data on https://insarmaps.miami.edu/

## Installation

The below install instructions have been tested on ubuntu 22.04. Installing on Windows/MacOS/other linux distributions will involve modifying the below commands to the equivalent ones on the target system.

1. Install tippecanoe: https://github.com/mapbox/tippecanoe
2. Install gdal: ```sudo apt-get install gdal-bin```
3. Install Mintpy: https://github.com/insarlab/MintPy/blob/main/docs/installation.md
4. Run `conda install -file environment.yml` to install non-python code (tippecanoe)
5. Run `pip install -r requirements.txt` to install remaining requirements

## Csv usage
### NOAA TRE Displacement Test Dataset

Download the test data;

```wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/sarvey_test.csv```
```wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/North_20162023.csv```
