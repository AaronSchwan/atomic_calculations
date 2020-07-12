"""
This is a bunch of clustering algorithms implimented for the use specifically
with data from the dimensionalReduction file


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
from sklearn import *
import numpy as np
import pandas as pd


def kmean(data:pd.DataFrame,number_of_clusters:int):
    """
    Simply uses sklearn k means package and returns a fit on the data

    kmean(data:pd.DataFrame,number_of_clusters:int)

    data = data that the kmeans class will be fit on
    number_of_clusters = how many clusters the kmeans fit will have

    returns a sklean kmeans fit
    """
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(data)

    return kmeans

def kmean_prediction(data:pd.DataFrame,kmeans)->pd.DataFrame:
    """
    This function will take a fitted kmeans and return the predictions by a pandas
    Dataframe

    kmean_prediction(data:pd.DataFrame,kmeans)

    data = data to have kmeans prediction on
    kmeans = fit kmeans class from sklearn

    returns a pandas dataframe with the label "kmean_prediction"
    """
    predictions = pd.DataFrame(data = kmeans.predict(k_data))
    predictions = predictions.rename({0: "kmean_prediction"}, axis='columns')
    return predictions

def kmean_fit_prediction(identification_numbering:pd.DataFrame,kmeans_data:pd.DataFrame,number_of_clusters:int)->pd.DataFrame:
    """
    This is useful for tracking through a progression of time. The function makes
    a kmeans fit on the data concecates the predictions with an id list allowing
    for a pandas merge on the id list with later data sets

    kmean_fit_prediction(identification_numbering:pd.DataFrame,kmeans_data:pd.DataFrame,number_of_clusters:int)->pd.DataFrame

    identification_numbering = pandas dataframe with the id column you want to track
    kmeans_data = data that will be kmeans fit
    number_of_clusters = how many kmeans clusters are desired

    returns a pandas dataframe with the column "labels"
    """
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(kmeans_data)
    predictions = pd.DataFrame(data = kmeans.predict(kmeans_data))
    predictions = predictions.rename({0: "labels"}, axis='columns')
    labels = pd.concat([identification_numbering, predictions], axis=1, sort=False)
    return labels
