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




class FittingPCA():
    """
    This class will fit a pca to the given data and files

    #####################################################################
    fitting pca valid calls:

    p = FittingPCA(pca_components,normalize,pca_columns,files,removed_values,filtered_values)

    p.pca #returns the trained pca
    p.normalize #returns if the pca used normalized data
    p.pca_components #returns the components the pca was trained for
    p.pca_columns #returns list of columns used to fit pca on
    p.serial #returns serial number of the pca data default val = None
    p.note #returns a note on the class as a list default val = None

    """

    def __init__(self,pca_components:int,normalize:bool,data:pd.DataFrame,serial:int=None,note:list=None):

        #saving conditions
        self.normalize = normalize
        self.pca_columns = data.columns.tolist()
        self.pca_components = pca_components
        self.serial = serial
        self.note = note
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

class TransformPCA():
    """
    this class will transform data with a pca fit

    #####################################################################
    transform valid calls:
    t = TransformData(data,pca_class,normalize)

    t.transform = returns the transform of the class
    t.normalize = returns if the transform was normalized before
    t.pca_serial = serial number from the FittingPCA() class
    t.pca_note = note from the FittingPCA() class
    t.serial = serial numbering, default val = None
    t.note = returns note on the transform class saved in a list, default val = None
    """

    def __init__(self,data:pd.DataFrame,pca_class:FittingPCA,normalize:bool,serial:int=None,note:list=None)->pd.DataFrame:
        #pca class conditions
        self.pca_serial = pca_class.serial
        self.pca_note = pca_class.note


        #transform conditions
        self.serial = serial
        self.note = note
        self.normalize = normalize

        #Geting transform
        data = lor.dumpFile(file_path).atoms
        data = data[pca_class.pca_columns]

        #transforming data
        #checking for correct columns
        data_columns = data.columns.tolist()
        pca_columns = pca_class.pca_columns

        if sorted(data_columns) == sorted(pca_columns):
            data_to_transform = data[pca_columns]#make the order the same as the pca class
            self.transform = self.transform_data(data_to_transform,pca_class.pca,normalize,pca_class.pca_components)
            del data_to_transform,data_columns,pca_columns

        else:
            raise ValueError('The columns of the data passed do not contain the same elements the FittingPCA() class was trained on')


    def transform_data(self,data,pca,normalize,pca_components):
        if normalize == True:
            data = pd.DataFrame(preprocessing.scale(data),columns = data.columns)#normalize data

        pca_data_array = pca.transform(data)

        titles = []
        for i in range(1,pca_components+1):
            titles.append(f"PC{i}")

        pca_data = pd.DataFrame(data=pca_data_array,  columns=titles)
        return pca_data
