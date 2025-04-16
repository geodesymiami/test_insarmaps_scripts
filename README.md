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
### NOAA TRE and SARvey Displacement Test Dataset

- Download the first test data;

```
wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/sarvey_test.csv
test_hdfeos5_2json_mbtiles.py sarvey_test.csv ./JSON_1
json_mbtiles2insarmaps.py --num-workers 3 -u insaradmin -p insaradmin --host 149.165.153.50 -P insarmaps -U insarmaps@insarmaps.com --json_folder ./JSON_1 --mbtiles_file ./JSON_1/sarvey_test.mbtiles

http://149.165.153.50/start/32.0269/49.3980/13.7305?flyToDatasetCenter=true&startDataset=sarvey_test
```
**OR,**

- Download the second test data;

```
wget http://149.165.154.65/data/HDF5EOS/epehlivanli/test_csv/North_20162023.csv
test_hdfeos5_2json_mbtiles.py North_20162023 ./JSON_2
json_mbtiles2insarmaps.py --num-workers 3 -u insaradmin -p insaradmin --host 149.165.153.50 -P insarmaps -U insarmaps@insarmaps.com --json_folder ./JSON_2 --mbtiles_file ./JSON_2/North_20162023.mbtiles

http://149.165.153.50/start/25.9479/-80.1186/15.0522?flyToDatasetCenter=true&startDataset=North_20162023
```

**NOTE!**
Before re-testing the same data;
```
insarmapsremove sarvey_test
insarmapsremove North_20162023
```
