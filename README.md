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

Download the first test data;

1. `wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/sarvey_test.csv`
2. Run `test_hdfeos5_2json_mbtiles.py ./test_csv/sarvey_test.csv ./test_csv/JSON_1` (Note: `./test_csv/JSON_1` is output directory which is created automatically, you don't need to create it)
3. Run `json_mbtiles2insarmaps.py --num-workers 3 -u insaradmin -p insaradmin --host 149.165.153.50 -P insarmaps -U insarmaps@insarmaps.com --json_folder ./test_csv/JSON_1 --mbtiles_file ./test_csv/OutDir/sarvey_test.mbtiles`

Download the second test data;

1. `wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/North_20162023.csv`
2. Run `test_hdfeos5_2json_mbtiles.py ./test_csv/North_20162023 ./test_csv/JSON_2` (Note: `./test_csv/JSON_2` is output directory which is created automatically, you don't need to create it) 
3. Run `json_mbtiles2insarmaps.py --num-workers 3 -u insaradmin -p insaradmin --host 149.165.153.50 -P insarmaps -U insarmaps@insarmaps.com --json_folder ./test_csv/JSON_2 --mbtiles_file ./test_csv/OutDir/North_20162023.mbtiles`
