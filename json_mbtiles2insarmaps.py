#! /usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Alfredo Terrero, 2016                            #
############################################################


import os
import sys
import argparse
import pickle
import numpy

from add_attribute_insarmaps import (
    InsarDatabaseController,
    InsarDatasetController,
)
from queue import Queue
from threading import Thread


dbUsername = "INSERT"
dbPassword = "INSERT"
dbHost = "INSERT"

# use pickle file to get unavco_name of dataset
def get_unavco_name(json_path):
    insarmapsMetadata = None
    fileName = json_path + "/metadata.pickle"

    with open(fileName, "rb") as file:
        insarmapsMetadata = pickle.load(file)

    return insarmapsMetadata["area"]

def upload_insarmaps_metadata(fileName):
    insarmapsMetadata = None

    with open(fileName, "rb") as file:
        insarmapsMetadata = pickle.load(file)

    area = insarmapsMetadata["area"]
    project_name = insarmapsMetadata["project_name"]
    mid_long = insarmapsMetadata["mid_long"]
    mid_lat = insarmapsMetadata["mid_lat"]
    country = insarmapsMetadata["country"]
    region = insarmapsMetadata["region"]
    chunk_num = insarmapsMetadata["chunk_num"]
    attribute_keys = insarmapsMetadata["attribute_keys"]
    attribute_values = insarmapsMetadata["attribute_values"]
    string_dates_sql = insarmapsMetadata["string_dates_sql"]
    decimal_dates_sql = insarmapsMetadata["decimal_dates_sql"]
    attributes = insarmapsMetadata["attributes"]
    needed_attributes = insarmapsMetadata["needed_attributes"]

    attributesController = InsarDatabaseController(dbUsername, dbPassword, dbHost, 'pgis')
    attributesController.connect()
    attributesController.create_area_table_if_not_exists()
    attributesController.insert_dataset_into_area_table(area, project_name,
    mid_long, mid_lat, country, region, chunk_num, attribute_keys,
    attribute_values, string_dates_sql, decimal_dates_sql)

    # put in attributes into standalone attributes table
    for k in attributes:
        v = attributes[k]
        # convert numpy.int64 objects to native python types otherwise psycopg2 can't upload to db
        # needed because we use pickle.HIGHEST_PROTOCOL to serialize with pickle now
        if isinstance(v, numpy.int64):
            v = v.item()

        if k in needed_attributes:
            attributesController.add_attribute(area, k, v)
        elif k == "plotAttributes":
            attributesController.add_plot_attribute(area, k, v)

    attributesController.close()

def worker(queue):
    # """Process files from the queue."""
    for args in iter(queue.get, None):
        try:
            command = args[0]
            file = args[1]

            res = os.system(command)

            if res != 0:
                msg = "Error inserting into the database."
                msg += " This is most often due to running out of Memory (RAM)"
                msg += ", or incorrect database credentials... quitting"
                sys.stderr.write(msg)
                os._exit(-1)

            print("Inserted " + file + " to db")

        except Exception as e: # catch exceptions to avoid exiting the
                               # thread prematurely
           print('%r failed: %s' % (args, e,))

def upload_json(folder_path, num_workers=1):
    global dbUsername, dbPassword, dbHost
    attributesController = InsarDatabaseController(dbUsername, dbPassword, dbHost, 'pgis')
    attributesController.connect()
    print("Clearing old dataset, if it is there")
    area_name = get_unavco_name(folder_path)
    try:
        attributesController.remove_dataset(area_name)
    except Exception as e:
        print(str(e))

    attributesController.close()

    # uploading metadata for area. this creates entry into area table.
    # we need this entry to get the db to generate an id for area which
    # we use to name the corresponding table for the dataset
    upload_insarmaps_metadata(folder_path + "/metadata.pickle")
    # create index
    print("Creating index on " + area_name)
    attributesController.connect()
    area_id = str(attributesController.get_dataset_id(area_name))
    attributesController.close()
    firstJsonFile = True

    q = Queue()
    threads = [Thread(target=worker, args=(q,)) for _ in range(num_workers)]
    for t in threads:
        t.daemon = True # threads die if the program dies
        t.start()

    for file in os.listdir(folder_path):
        # insert json file to pgsql using ogr2ogr
        file_extension = file.split(".")[1]
        if file_extension == "json":
            command = 'ogr2ogr -append -f "PostgreSQL" PG:"dbname=pgis ' + \
                      ' host=' + dbHost + ' user=' + dbUsername + ' password=' + dbPassword + \
                      '" --config PG_USE_COPY YES -nln ' + area_id + ' ' + folder_path + '/' + file
            # only provide layer creation options if this is the first file
            if firstJsonFile:
                command = 'ogr2ogr -lco LAUNDER=NO -append -f "PostgreSQL" PG:"dbname=pgis ' + \
                          ' host=' + dbHost + ' user=' + dbUsername + ' password=' + dbPassword + \
                          '" --config PG_USE_COPY YES -nln ' + area_id + ' ' + folder_path + '/' + file
                firstJsonFile = False
                # upload first one with layer creation options prior
                # to spawning processes to upload the rest
                res = os.system(command)
                print("Inserted " + file + " to db")

                if res != 0:
                    msg = "Error inserting into the database."
                    msg += " This is most often due to running out of Memory (RAM)"
                    msg += ", or incorrect database credentials... quitting"
                    sys.stderr.write(msg)
                    os._exit(-1)
            else:
              q.put_nowait([command, file])

    for _ in threads: q.put_nowait(None) # signal no more files
    for t in threads: t.join() # wait for completion

    attributesController.connect()
    print("Indexing table")
    attributesController.index_table_on(area_id, "p", None)
    print("Clustering table")
    attributesController.cluster_table_using(area_id, area_id + "_p_idx")
    attributesController.close()

def build_parser():
    dbHost = "insarmaps.rsmas.miami.edu"
    parser = argparse.ArgumentParser(description='Convert a Unavco format HDF5 file for ingestion into insarmaps.')
    parser.add_argument("--num-workers", help="Number of simultaneous processes to run for ingest.", required=False, default=1, type=int)
    parser.add_argument("--json_folder", help="folder containing json to upload.", required=False)
    parser.add_argument("json_folder_positional", help="folder containing json to upload.", nargs="?")
    parser.add_argument("-U", "--server_user", required=False, 
        help="username for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--remove",
        help="UNAVCO name of dataset to remove from insarmaps website", required=False)
    parser.add_argument("--list",
        help="List datasets currently on the insarmaps website", required=False, action="store_true")
    parser.add_argument("-P", "--server_password", required=False,
        help="password for the insarmaps server (the machine where the tileserver and http server reside)")
    parser.add_argument("--mbtiles_file", help="mbtiles file to upload", required=False)
    parser.add_argument("mbtiles_file_positional",
        help="mbtiles file to upload, as a positional argument", nargs="?")

    required = parser.add_argument_group("required arguments")
    required.add_argument("-u", "--user", help="username for the insarmaps database", required=True)
    required.add_argument("-p", "--password", help="password for the insarmaps database", required=True)
    required.add_argument("--host", default=dbHost, help="postgres DB URL for insarmaps database", required=True)

    return parser

def main():
    global dbUsername, dbPassword, dbHost
    parser = build_parser()
    parseArgs = parser.parse_args()
    dbUsername = parseArgs.user
    dbPassword = parseArgs.password
    dbHost = parseArgs.host

    if parseArgs.json_folder:
        print("Uploading json chunks...")
        upload_json(parseArgs.json_folder, parseArgs.num_workers)
    elif parseArgs.json_folder_positional:
        print("Uploading json chunks....")
        upload_json(parseArgs.json_folder_positional, parseArgs.num_workers)

    dbController = InsarDatasetController(dbUsername,
                                         dbPassword,
                                         dbHost,
                                         'pgis',
                                         parseArgs.server_user,
                                         parseArgs.server_password)

    if parseArgs.mbtiles_file or parseArgs.mbtiles_file_positional:
        if not parseArgs.server_user or not parseArgs.server_password:
            sys.stderr.write("Error: credentials for the insarmaps server not provided")
        elif parseArgs.mbtiles_file:
            print("Uploading mbtiles...")
            if os.path.isfile(parseArgs.mbtiles_file):
                dbController.upload_mbtiles(parseArgs.mbtiles_file)
            else:
                print(parseArgs.mbtiles_file + " doesn't exist")
        else:
            print("Uploading mbtiles....")
            if os.path.isfile(parseArgs.mbtiles_file_positional):
                dbController.upload_mbtiles(parseArgs.mbtiles_file_positional)
            else:
                print(parseArgs.mbtiles_file_positional + " doesn't exist")

    if parseArgs.remove:
        if not parseArgs.server_user or not parseArgs.server_password:
            sys.stderr.write("Error: credentials for the insarmaps server not provided")
        else:
            print("Trying to remove %s" % (parseArgs.remove))
            dbController.connect()
            try:
                dbController.remove_dataset(parseArgs.remove)
                dbController.remove_mbtiles(parseArgs.remove + ".mbtiles")
                print("Successfully removed %s" % (parseArgs.remove))
            except Exception as e:
                print(str(e))

            dbController.close()


    if parseArgs.list:
        dbController.connect()
        dbController.list_dataset_names()
        dbController.close()

if __name__ == '__main__':
    main()
