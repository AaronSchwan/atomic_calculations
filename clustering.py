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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import *
################################################################################
#KMEAN Clustering

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

################################################################################
#GMM (Gaussian Mixture Model) Clustering
def GMM_get_num_clusters(data:pd.DataFrame,max_num:int = 10):
    n_estimators = np.arange(2,max_num)
    clfs = [GMM(n_components=n).fit(data) for n in n_estimators]
    bics = [clf.bic(data) for clf in clfs]
    aics = [clf.aic(data) for clf in clfs]

    n_components = round((bics.index(min(bics)) + aics.index(min(aics)))/2)

    return n_components

def GMM_fit_prediction(data:pd.DataFrame,num_components:int = None):
    if num_components == None:
        num_components = GMM_get_num_clusters(data)

    gmm = GMM(n_components=num_components).fit(data)
    predictions = pd.DataFrame(data = gmm.predict(data))
    predictions = predictions.rename({0: "gmm_prediction"}, axis='columns')
    return predictions

def GMM_fit(data:pd.DataFrame,num_components:int=None):
    if num_components == None:
        num_components = GMM_get_num_clusters(data)

    gmm = GMM(n_components=num_components).fit(data)
    return gmm

def GMM_prediction(data:pd.DataFrame,gmm):
    predictions = pd.DataFrame(data = gmm.predict(data))
    predictions = predictions.rename({0: "gmm_prediction"}, axis='columns')
    return predictions
################################################################################
#DBSCAN (Density-Based Spatial Clustering)

################################################################################
#FINCH (First Integer Neighbor Clustering)
