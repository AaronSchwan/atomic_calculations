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

#importing non-default programs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
from sklearn.pipeline import Pipeline

#custom imports
import basicFunctions as bf
import LammpsOvitoReader as lor


class TransformData():
    """
    this class will transform data with a pca fit

    #####################################################################
    transform valid calls:
    t = TransformData(file_path,pca_class,normalize)

    t.transform #returns the transform of the class
    t.file_name #returns the name of the file that was transformed
    t.pca_serial #returns the serial number of the PCA used to transform
    t.normalize #returns if the transform was normalized before
    """

    def __init__(self,file_path,pca_class,normalize_in):
        self.file_name = ntpath.basename(file_path)
        self.pca_serial = pca_class.serial
        self.normalize = normalize_in

        #Geting data
        data = lor.dumpFile(file_path).atoms
        data = data[pca_class.pca_columns]

        #transforming data
        self.transform = self.transform_data(data,pca_class.pca,normalize_in,pca_class.pca_components)
    def transform_data(self,data,pca,normalize,pca_components):
        if normalize == True:
            data = pd.DataFrame(preprocessing.scale(data),columns = data.columns)#normalize data

        pca_data_array = pca.transform(data)

        titles = []
        for i in range(1,pca_components+1):
            titles.append(f"PC{i}")

        pca_data = pd.DataFrame(data=pca_data_array,  columns=titles)
        return pca_data


class FittingPCA():
    """
    This class will fit a pca to the given data and files

    #####################################################################
    fitting pca valid calls:

    p = FittingPCA(pca_components,normalize,pca_columns,files,removed_values,filtered_values)

    p.pca #returns the trained pca
    p.file_names #returns names of files used to fit pca
    p.normalize #returns if the pca used normalized data
    p.pca_components #returns the components the pca was trained for
    p.pca_columns #returns list of columns used to fit pca on
    p.filtered_values #returns list of filtered data vals
    p.removed_values #returns list of removed data vals
    p.serial #returns serial number of the pca data

    """

    def __init__(self,pca_components_in:int,normalize_in:bool,pca_columns_in:list,files_in:list,removed_values_in:dict,filtered_values_in:dict,serial_in:str=None):


        self.pca_columns = sorted(pca_columns_in)

        #sorting names
        file_names = []
        for f in files_in:
            file_names.append(ntpath.basename(f))


        #importing data
        data = self.import_data(files_in,self.pca_columns,removed_values_in,filtered_values_in)
        #saving conditions
        self.file_names = sorted(file_names)
        self.normalize = normalize_in
        self.pca_components = pca_components_in
        self.filtered_values = filtered_values_in
        self.removed_values = removed_values_in
        self.serial = serial_in
        #calculating pca
        self.pca =  self.computing_PCA(pca_components_in,data,normalize_in)

        del data

    def computing_PCA(self,pca_components,data,normalize):
        fitted_pca = PCA(n_components=pca_components)#making pca structure

        if normalize == True:
            #if normalizeing needed
            data = pd.DataFrame(preprocessing.scale(data),columns = data.columns)

        fitted_pca.fit(data)#Fitting PCA

        return fitted_pca


    def import_data(self,files,pca_columns,removed_values,filtered_values):

        for ind,file in enumerate(files):
            gc.collect()#clear what it can

            if ind == 0:
                #Setting base dataframe
                data = lor.dumpFile(file).atoms

            else:
                #appending to base dataframe
                data_grab = lor.dumpFile(file).atoms
                data = data.append(data_grab)

        #cleaning dataframe
        if filtered_values != {}:
            data = bf.filtering_data(data,filtered_values)

        if removed_values != {}:
            data = bf.removing_data(data,removed_values)

        data = data[pca_columns]

        return data
