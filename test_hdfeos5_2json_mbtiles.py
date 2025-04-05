#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import os
import sys
import argparse
import pickle
import json
import time
from datetime import date
import math
import geocoder
import numpy as np
from pathlib import Path

from mintpy.objects import HDFEOS
from mintpy.mask import mask_matrix
from mintpy.utils import utils as ut
import h5py
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing import Pool
from multiprocessing import Value

import pandas as pd
from datetime import datetime

chunk_num = Value("i", 0)
# ex: python Converter_unavco.py Alos_SM_73_2980_2990_20070107_20110420.h5

# This script takes a UNAVCO format timeseries h5 file, converts to mbtiles, 
# and sends to database which allows website to make queries and display data
# ---------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------
# returns a dictionary of datasets that are stored in memory to speed up h5 read process
def get_date(date_string): 
    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])
    return date(year, month, day)


# ---------------------------------------------------------------------------------------
# takes a date and calculates the number of days elapsed in the year of that date
# returns year + (days_elapsed / 365), a decimal representation of the date necessary
# for calculating linear regression of displacement vs time
def get_decimal_date(d):
    start = date(d.year, 1, 1)
    return abs(d-start).days / 365.0 + d.year

def region_name_from_project_name(project_name):
    track_index = project_name.find('T')

    return project_name[:track_index]

needed_attributes = {
    "prf", "first_date", "mission", "WIDTH", "X_STEP", "processing_software",
    "wavelength", "processing_type", "beam_swath", "Y_FIRST", "look_direction",
    "flight_direction", "last_frame", "post_processing_method", "min_baseline_perp"
    "unwrap_method", "relative_orbit", "beam_mode", "LENGTH", "max_baseline_perp",
    "X_FIRST", "atmos_correct_method", "last_date", "first_frame", "frame", "Y_STEP", "history",
    "scene_footprint", "data_footprint", "downloadUnavcoUrl", "referencePdfUrl", "areaName", "referenceText",
    "REF_LAT", "REF_LON", "CENTER_LINE_UTC", "insarmaps_download_flag", "mintpy.subset.lalo"
}

def serialize_dictionary(dictionary, fileName):
    with open(fileName, "wb") as file:
        pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
    return

def get_attribute_or_remove_from_needed(needed_attributes, attributes, attribute_name):
    val = None

    try:
        val = attributes[attribute_name]
    except:
        needed_attributes.remove(attribute_name)

    return val

def generate_worker_args(decimal_dates, timeseries_datasets, dates, json_path, folder_name, chunk_size, lats, lons, num_columns, num_rows):
    num_points = num_columns * num_rows

    worker_args = []
    start = 0
    end = 0
    idx = 0
    for i in range(num_points // chunk_size):
        start = idx * chunk_size
        end = (idx + 1) * chunk_size
        if end > num_points:
            end = num_points
        args = [decimal_dates, timeseries_datasets, dates, json_path, folder_name, (start, end - 1), num_columns, num_rows, lats, lons]
        worker_args.append(tuple(args))
        idx += 1

    if num_points % chunk_size != 0:
        start = end
        end = num_points
        args = [decimal_dates, timeseries_datasets, dates, json_path, folder_name, (start, end - 1), num_columns, num_rows, lats, lons]
        worker_args.append(tuple(args))

    return worker_args

def create_json(decimal_dates, timeseries_datasets, dates, json_path, folder_name, work_idxs, num_columns, num_rows, lats=None, lons=None):
    global chunk_num
    # create a siu_man array to store json point objects
    siu_man = []
    displacement_values = []
    displacements = '{'
    # np array of decimal dates, x parameter in linear regression equation
    x = decimal_dates
    A = np.vstack([x, np.ones(len(x))]).T
    y = []
    point_num = work_idxs[0]
    # iterate through h5 file timeseries
    for (row, col), value in np.ndenumerate(timeseries_datasets[dates[0]]):
        cur_iter_point_num = row * num_columns + col
        if cur_iter_point_num < work_idxs[0]:
            continue
        elif cur_iter_point_num > work_idxs[1]:
            break

        longitude = float(lons[row][col])
        latitude = float(lats[row][col])
        displacement = float(value) 
        # if value is not equal to naN, create a new json point object and append to siu_man array
        if not math.isnan(displacement):
            # get displacement values for all the dates into array for json and string for pgsql
            for datei in dates:
                displacement = timeseries_datasets[datei][row][col]
                displacements += (str(displacement) + ",")
                displacement_values.append(float(displacement))
            displacements = displacements[:len(displacements) - 1] + '}'

            # np array of displacement values, y parameter in linear regression equation
            y = displacement_values

            # y = mx + c -> we want m = slope of the linear regression line 
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            data = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},    
            "properties": {"d": displacement_values, "m": m, "p": point_num}
            }   

            siu_man.append(data)

            # clear displacement array for json and the other string for dictionary, for next point
            displacement_values = []
            displacements = '{'
            point_num += 1

    if len(siu_man) > 0:
        chunk_num_val = -1
        with chunk_num.get_lock():
            chunk_num_val = chunk_num.value
            chunk_num.value += 1

        make_json_file(chunk_num_val, siu_man, dates, json_path, folder_name)
        siu_man = []

# ---------------------------------------------------------------------------------------
# convert h5 file to json and upload it. folder_name == unavco_name
def convert_data(attributes, decimal_dates, timeseries_datasets, dates, json_path, folder_name, lats=None, lons=None, num_workers=1):

    project_name = attributes["PROJECT_NAME"]
    region = region_name_from_project_name(project_name)
    # get the attributes for calculating latitude and longitude
    x_step, y_step, x_first, y_first = 0, 0, 0, 0
    if high_res_mode(attributes):
        needed_attributes.remove("X_STEP")
        needed_attributes.remove("Y_STEP")
        needed_attributes.remove("X_FIRST")
        needed_attributes.remove("Y_FIRST")
    else:
        x_step = float(attributes["X_STEP"])
        y_step = float(attributes["Y_STEP"])
        x_first = float(attributes["X_FIRST"])
        y_first = float(attributes["Y_FIRST"])

    num_columns = int(attributes["WIDTH"])
    num_rows = int(attributes["LENGTH"])
    print("columns: %d" % num_columns)
    print("rows: %d" % num_rows)
    if lats is None and lons is None:
        lats, lons = ut.get_lat_lon(attributes, dimension=1)

    CHUNK_SIZE = 20000
    process_pool = Pool(num_workers)
    process_pool.starmap(create_json, generate_worker_args(decimal_dates, timeseries_datasets, dates, json_path, folder_name, CHUNK_SIZE, lats, lons, num_columns, num_rows))
    process_pool.close()

    # dictionary to contain metadata needed by db to be written to a file
    # and then be read by json_mbtiles2insarmaps.py
    insarmapsMetadata = {}
    # calculate mid lat and long of dataset - then use google python lib to get country
    # technically don't need the else since we always use lats and lons arrays now
    
    #below part removed for now ,csv
    #if high_res_mode(attributes):
    #    num_rows, num_columns = lats.shape
    #    mid_long = float(lons[num_rows // 2][num_columns // 2])
    #    mid_lat = float(lats[num_rows // 2][num_columns // 2])
    #else:
    #    mid_long = x_first + ((num_columns/2) * x_step)
    #    mid_lat = y_first + ((num_rows/2) * y_step)
    mid_lat = float(np.nanmean(lats))
    mid_long = float(np.nanmean(lons))

    country = "None"
    try:
        g = geocoder.google([mid_lat,mid_long], method='reverse', timeout=60.0)
        country = str(g.country_long)
    except Exception:
        sys.stderr.write("timeout reverse geocoding country name")

    area = folder_name

    # for some reason pgsql only takes {} not [] - format date arrays and attributes to be inserted to pgsql
    string_dates_sql = '{'
    for k in dates:
        string_dates_sql += (str(k) + ",")
    string_dates_sql = string_dates_sql[:len(string_dates_sql) - 1] + '}'

    decimal_dates_sql = '{'
    for d in decimal_dates:
        decimal_dates_sql += (str(d) + ",")
    decimal_dates_sql = decimal_dates_sql[:len(decimal_dates_sql) - 1] + '}'
    # add keys and values to area table. TODO: this will be removed eventually
    # and all attributes will be put in extra_attributes table
    attribute_keys = '{'
    attribute_values = '{'
    max_digit = max([len(key) for key in list(needed_attributes)] + [0])
    for k in attributes:
        v = attributes[k]
        if k in needed_attributes:
            print('{k:<{w}}     {v}'.format(k=k, w=max_digit, v=v))
            attribute_keys += (str(k) + ",")
            attribute_values += (str(v) + ',')
    attribute_keys = attribute_keys[:len(attribute_keys)-1] + '}'
    attribute_values = attribute_values[:len(attribute_values)-1] + '}'

    # write out metadata to json file
    insarmapsMetadata["area"] = area
    insarmapsMetadata["project_name"] = project_name
    insarmapsMetadata["mid_long"] = mid_long
    insarmapsMetadata["mid_lat"] = mid_lat
    insarmapsMetadata["country"] = country
    insarmapsMetadata["region"] = region
    insarmapsMetadata["chunk_num"] = 1
    insarmapsMetadata["attribute_keys"] = attribute_keys
    insarmapsMetadata["attribute_values"] = attribute_values
    insarmapsMetadata["string_dates_sql"] = string_dates_sql
    insarmapsMetadata["decimal_dates_sql"] = decimal_dates_sql
    insarmapsMetadata["attributes"] = attributes
    insarmapsMetadata["needed_attributes"] = needed_attributes
    metadataFilePath = json_path + "/metadata.pickle" 
    serialize_dictionary(insarmapsMetadata, metadataFilePath)
    return


# ---------------------------------------------------------------------------------------
# create a json file out of siu man array
# then put json file into directory named after the h5 file
def make_json_file(chunk_num, points, dates, json_path, folder_name):
    chunk = "chunk_" + str(chunk_num) + ".json"
    json_file = open(json_path + "/" + chunk, "w")
    json_features = [json.dumps(feature) for feature in points]
    string_json = '\n'.join(json_features)
    json_file.write("%s" % string_json)
    json_file.close()

    print("converted chunk " + str(chunk_num))
    return chunk

def high_res_mode(attributes):
    high_res = False # default
    try:
        x_step = attributes["X_STEP"]
        y_step = attributes["Y_STEP"]
    except:
        high_res = True # one or both not there, so we are high res

    return high_res

# ---------------------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description='Convert a Unavco format H5 file for ingestion into insarmaps.', epilog="This program will create temporary json chunk files which, when concatenated together, comprise the whole dataset. Tippecanoe is used for concatenating these chunk files into the mbtiles file which describes the whole dataset.")
    parser.add_argument("--num-workers", help="Number of simultaneous processes to run for ingest.", required=False, default=1, type=int)
    required = parser.add_argument_group("required arguments")
    required.add_argument("file", help="unavco file to ingest")
    required.add_argument("outputDir", help="directory to place json files and mbtiles file")

    return parser

def read_from_hdfeos5_file(file_name):
    # read data from hdfeos5 file
    should_mask = True

    path_name_and_extension = os.path.basename(file_name).split(".")
    path_name = path_name_and_extension[0]

    # use h5py to open specified group(s) in the h5 file 
    # then read datasets from h5 file into memory for faster reading of data
    he_obj = HDFEOS(file_name)
    he_obj.open(print_msg=False)
    displacement_3d_matrix = he_obj.read(datasetName='displacement')
    mask = he_obj.read(datasetName='mask')
    if should_mask:
        print("Masking displacement")
        displacement_3d_matrix = mask_matrix(displacement_3d_matrix, mask)
    del mask

    print("Creating shared memory for multiple processes")
    shm = shared_memory.SharedMemory(create=True, size=displacement_3d_matrix.nbytes)
    shared_displacement_3d_matrix = np.ndarray(displacement_3d_matrix.shape, dtype=displacement_3d_matrix.dtype, buffer=shm.buf)
    shared_displacement_3d_matrix[:] = displacement_3d_matrix[:]
    del displacement_3d_matrix
    displacement_3d_matrix = shared_displacement_3d_matrix

    dates = he_obj.dateList
    attributes = dict(he_obj.metadata)

    decimal_dates = []

    # read datasets in the group into a dictionary of 2d arrays and intialize decimal dates
    timeseries_datasets = {}
    num_date = len(dates)
    for i in range(num_date):
        timeseries_datasets[dates[i]] = np.squeeze(displacement_3d_matrix[i, :, :])
        d = get_date(dates[i])
        decimal = get_decimal_date(d)
        decimal_dates.append(decimal)
    del displacement_3d_matrix

    path_list = path_name.split("/")
    folder_name = path_name.split("/")[len(path_list)-1]

    # read lat and long. MintPy doesn't seem to support this yet, so we use the raw
    # h5 file object
    f = h5py.File(he_obj.file, "r")
    lats = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["latitude"])
    lons = np.array(f["HDFEOS"]["GRIDS"]["timeseries"]["geometry"]["longitude"])

    return attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm

def read_from_csv_file(file_name):
    # read data from csv file to be done by Emirhan
    # the shared memory shm is confusing. it may also works without but be careful about returning or not returning shm.

    df = pd.read_csv(file_name)

    #extract values
    lats = df["Latitude"].values
    lons = df["Longitude"].values
    vel_v = df["VEL_V"].values
    v_stdev = df["V_STDEV_V"].values

    #date columns
    time_cols = [col for col in df.columns if col.isdigit()]
    time_cols.sort()

    #3D time-series array (time, y, x)
    num_points = len(df)
    num_dates = len(time_cols)
    timeseries_data = df[time_cols].values

    #convert 2D slices for each date
    num_rows = int(np.sqrt(num_points))
    num_cols = int(np.ceil(num_points / num_rows))
    padded = np.full((num_cols * num_rows, num_dates), np.nan)
    padded[:num_points, :] = timeseries_data

    reshaped = padded.reshape((num_rows, num_cols, num_dates)).transpose(2, 0, 1)

    #dates
    dates = time_cols
    decimal_dates = [get_decimal_date(get_date(d)) for d in dates]
    timeseries_datasets = {d: reshaped[i, :, :] for i, d in enumerate(dates)}

    #required metadata
    attributes = {
        "PROJECT_NAME": "CSV_IMPORT",
        "WIDTH": str(num_cols),
        "LENGTH": str(num_rows),
        "X_STEP": "1.0",
        "Y_STEP": "1.0",
        "X_FIRST": str(min(lons)),
        "Y_FIRST": str(min(lats)),
    }

    #reshape lat/lons
    padded_lats = np.full(num_cols * num_rows, np.nan)
    padded_lats[:num_points] = lats
    lats_grid = padded_lats.reshape((num_rows, num_cols))

    padded_lons = np.full(num_cols * num_rows, np.nan)
    padded_lons[:num_points] = lons
    lons_grid = padded_lons.reshape((num_rows, num_cols))

    folder_name = os.path.basename(file_name).split(".")[0]

    #shm none
    shm = None

    print(" Check: read_from_csv_file output")
    print(" - timeseries_datasets keys:", list(timeseries_datasets.keys())[:3], "...")
    print(" - sample slice shape:", timeseries_datasets[dates[0]].shape)
    print(" - lat_grid shape:", lats_grid.shape)
    print(" - lon_grid shape:", lons_grid.shape)
    print(" - Number of dates:", len(dates))
    print(" - Attributes:", attributes)

    return attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats_grid, lons_grid, shm
# ---------------------------------------------------------------------------------------
# START OF EXECUTABLE
# ---------------------------------------------------------------------------------------
def main():
    parser = build_parser()
    parseArgs = parser.parse_args()
    file_name = parseArgs.file
    output_folder = parseArgs.outputDir

    try: # create path for output
        os.mkdir(output_folder)
    except:
        print(output_folder + " already exists")

    file_path = Path(file_name)
    
    # start clock to track how long conversion process takes
    start_time = time.perf_counter()
    
    if file_path.suffix.lower() == ".he5":
        attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm = read_from_hdfeos5_file(file_name)
    elif file_path.suffix.lower() == ".csv":
        attributes, decimal_dates, timeseries_datasets, dates, folder_name, lats, lons, shm = read_from_csv_file(file_name)
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist or has not .he5 or .csv as extension.")   

    # read and convert the datasets, then write them into json files and insert into database
    convert_data(attributes, decimal_dates, timeseries_datasets, dates, output_folder, folder_name, lats, lons, parseArgs.num_workers)
    del lats
    del lons
    if shm is not None:
        shm.close()
        shm.unlink()

    # run tippecanoe command to get mbtiles file
    os.chdir(os.path.abspath(output_folder))
    cmd = None
    if high_res_mode(attributes):
        cmd = "tippecanoe *.json -P -l chunk_1 -x d -pf -pk -o " + folder_name + ".mbtiles 2> tippecanoe_stderr.log"
    else:
        cmd = "tippecanoe *.json -P -l chunk_1 -x d -pf -pk -Bg -d9 -D12 -g12 -r0 -o " + folder_name + ".mbtiles 2> tippecanoe_stderr.log"

    print("Now running tippecanoe with command %s" % cmd)
    os.system(cmd)
    # ---------------------------------------------------------------------------------------
    # check how long it took to read h5 file data and create json files
    end_time =  time.perf_counter()
    print(("time elapsed: " + str(end_time - start_time)))
    return

# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
