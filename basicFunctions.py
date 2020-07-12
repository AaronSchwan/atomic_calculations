"""
###############################################################################
###############################################################################
author: Aaron Schwan
email: schwanaaron@gmail.com
github: https://github.com/AaronSchwan
###############################################################################
###############################################################################

"""

#Imported by default
import sys
import os
import gc
import pickle
import platform
import warnings
import pkg_resources
import subprocess
import ntpath


#non-default imports
import pandas as pd
def pull_folder_contents(folder_path:str,file_type:str,naming_conv:str = None):
    """
    This function finds all files within in a folder following a certain filetype
    as well as a naming convention. this is done by a simple look for file_type
    in string

    #####################################################################
    pull_folder_contents(folder_path,file_type,naming_conv)

    takes in:
    folder_path = path to folder you want contents of(Unicode Str)
    file_type = Type of file you want to select
    nameing_conv = convention that must be present in name

    file_type can be selected as numeric which will test if the extension is a
    integer instead of a destinct file type

    returns:
    list of all file names matching description
    """
    temp_contents = os.listdir(folder_path)

    files = []
    #for dump files
    if naming_conv != None:
        if file_type == "numeric":
            for tc in temp_contents:
                if naming_conv in tc:
                    file = tc.split(".")
                    ext = file[len(file)-1]
                    try:
                        int(ext)
                        files.append(tc)
                    except:
                        pass;
        else:
            for tc in temp_contents:
                if file_type in tc and naming_conv in tc:
                    files.append(tc)
    else:
        if file_type == "numeric":
            for tc in temp_contents:
                file = tc.split(".")
                ext = file[len(file)-1]
                try:
                    int(ext)
                    files.append(tc)
                except:
                    pass;
        else:
            for tc in temp_contents:
                if file_type in tc:
                    files.append(tc)
    return files


def removing_data(data,removed_values:dict):
    """
    This function takes in a pandas datframe and removes rows based on column values
    #####################################################################
    removing_data(data,removed_values:dict)

    takes in:
    removed_values = values to be removed (Dictionary)
    {"Column1":[Value1a,Value1b],"Column2":[Value2a,Value2b]}

    data = pandas dataframe

    returns:
    pandas dataframe minus the requesit values
    """
    columns = removed_values.keys()

    for col in columns:
        values = removed_values[col]
        for ind,val in enumerate(values):
            data = data[eval(f"data.{col}") != values[ind]]
    return data

def filtering_data(data,filtered_values:dict):
    """
    This function takes in a pandas datframe and removes rows based on column values
    if they are not in the range of high and low this includes the highs and lows

    #####################################################################

    filtering_data(data,filtered_values)

    takes in:
    data = pandas dataframe
    filtered_values = dictionary to filter (dict)
    {"Column1":[high1,low1],"Column2":[high2,low2]}

    returns:
    pandas dataframe filtered

    """
    columns = filtered_values.keys()

    for col in columns:
        values = filtered_values[col]
        high = values[0]
        low = values[1]


        data = data[(data[col] <= high) & (data[col] >= low)]
    return data


def class_to_bianary(function_class,file_path):
    """
    This takes a class and the filepath and saves a bianary file with pickle

    class_to_bianary(function_class,file_path)

    """
    with open(file_path, 'wb') as output:
        pickle.dump(function_class, output, pickle.HIGHEST_PROTOCOL)

def bianary_to_class(file_path):
    """
    This reads a bianary file that has been pickled and converts it to a class

    bianary_to_class(file_path)
    """
    with open(file_path, 'rb') as input:
        data = pickle.load(input)
    return data
