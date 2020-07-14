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
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer

#custom imports
import basicFunctions as bf
import LammpsOvitoReader as lor




class FittingPCA():
    """
    This class will fit a pca to the given data and files

    #####################################################################
    fitting pca valid calls:

    p = FittingPCA(pca_components:int,normalize:bool,data:pd.DataFrame,serial:int=None,note:list=None)

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
        self.pca =  self.computing_PCA(pca_components,data,normalize)

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
    t = TransformData(data:pd.DataFrame,pca_class:FittingPCA,normalize:bool,serial:int=None,note:list=None)

    t.transform = returns the transform of the class
    t.normalize = returns if the transform was normalized before
    t.pca_serial = serial number from the FittingPCA() class
    t.pca_note = note from the FittingPCA() class
    t.serial = serial numbering, default val = None
    t.note = returns note on the transform class saved in a list, default val = None
    """

    def __init__(self,data:pd.DataFrame,pca_class:FittingPCA,normalize:bool,serial:int=None,note:list=None):
        #pca class conditions
        self.pca_serial = pca_class.serial
        self.pca_note = pca_class.note


        #transform conditions
        self.serial = serial
        self.note = note
        self.normalize = normalize


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

def bartlett_test(data):
    """
    sphericity checks whether or not the observed variables intercorrelate

    if this is found to be statistically insignificant factor analysis should not
    be used
    """

    chi_square_value,p_value=calculate_bartlett_sphericity(data)
    return chi_square_value,p_value

def kaiser_meyer_olkin_test(data):
    """
    measures the suitability for Factor analysis

    values less than .6 are considered inadequete range 0-1
    """
    kmo_all,kmo_model=calculate_kmo(data)
    return kmo_all,kmo_model

def skree_data_factor_analysis(data,factor_num, rotation_val=None):
    """
    this allows for a skree plot of the data relating to factor analysis to be made
    the number of values greator than 1 should be the number of factors
    """
    fa = FactorAnalyzer()
    fa.set_params(n_factors=factor_num, rotation=rotation_val)
    fa.fit(data)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    return ev



def factor_analysis(data:pd.DataFrame,rotation_val:str,factor_num:int = 0,adequacy_tests:bool=True):
    #checks and warnings
    if adequacy_tests == True:
        chi_square_value,p_value =calculate_bartlett_sphericity(data)
        kmo_all,kmo_model=calculate_kmo(data)

        if p_value > .05:
            warnings.warn(f"The p_value by the bartlett test is {p_value}. Factor analysis is not recommended")
        if kmo_model < .6:
            warnings.warn(f"The kaiser meyer olklin test returns a model score of {kmo_model}. Factor analysis is not recommended")

        del chi_square_value,p_value,kmo_all,kmo_model

    #find facor_num if unspecified
    if factor_num == 0:
        ev = skree_data_factor_analysis(data, len(data.columns.tolist()))
        factor_num = len([i for i in ev if i > 1])

    #performing factor analysis
    fa = FactorAnalyzer()
    fa.set_params(n_factors=factor_num, rotation=rotation_val)
    fa.fit(data)

    return fa
