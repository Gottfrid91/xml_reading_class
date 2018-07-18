from input_data import *
from model import *
import os
import numpy as np
import xml.etree.cElementTree as et
import pandas as pd
import xml_data_class as xdc
import depth_vector as dv
from scipy.stats import mode

def get_study_xml(study_path):
    '''
    :program info: this function loads the xml of a study and retrieves the OCT and LOCALIZER dimensions, scales and
    positional arguments of the OCT's with respect to the LOCALIZER. The program also asserts that the xml file is not
    corrupt with regards to the LOCALIZER dim, positional arguments. The program makes the following corrections:
    1. if any positional argument is negative, it is set to 0
    2. if the pixel dimension exceeds the allowed dimensions after translation form mm to pixel, the pixel dimension
        is set to 767 (max)
    3. If the xml contains only partial positional or dimensional information, i.e. contains missing values,
        then the missing values are filles using linear interpolation
    :param study_path: Path to the study dir currently in progress
    :return: the grid with dimension as LOCALIZER, and the OCT_pos_pixel, im_pd xml data frame
    '''
    stop_function = None
    #instatiate and tree parser and retrieve the im_pd from the xml class
    tree = et.ElementTree()
    tree.parse(study_path)
    root = tree.getroot()
    data = xdc.xml_data(root, tree)
    im_pd = data.get_image_table()

    # start retrieving the marked depth grid and y, x indices
    LOC_dim = im_pd[im_pd["Image_type"] == "LOCALIZER"][["Width", "Height"]]
    LOC_scale = im_pd[im_pd["Image_type"] == "LOCALIZER"][["scaleX", "scaleY"]]
    OCT_dim = im_pd[im_pd["Image_type"] == "OCT"][["Width", "Height"]]

    if LOC_dim.empty == True:
        print("The xml files does not contain the Volume data for patient: {}".format(patient))
        stop_function = "yes"

    #get dim of OCT image
    x_dim = int(LOC_dim.values[0][1])
    y_dim = int(LOC_dim.values[0][0])

    #create grid for depth map
    grid = np.zeros([y_dim, x_dim])

    #get localizer scale
    x_scale = float(LOC_scale.values[0][0])
    y_scale = float(LOC_scale.values[0][1])

    # convert positional arguments from mm to pixels
    OCT_pos_mm = im_pd[im_pd["Image_type"] == "OCT"][["startx_pos", "starty_pos", "endx_pos", "endy_pos"]] \
        .astype("float32")
    # interpolate possible Nan values
    OCT_pos_mm = OCT_pos_mm.interpolate(limit_direction='both')

    # Set negative values from corrupt files to 0
    num = OCT_pos_mm._get_numeric_data()
    num[num < 0] = 0

    # if all position values are Nan#s, then disregard the study
    if im_pd[im_pd["Image_type"] == "OCT"][["startx_pos"]].isnull().all().values[0] == True:
        print("Startx is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["starty_pos"]].isnull().all().values[0] == True:
        print("Starty is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["endx_pos"]].isnull().all().values[0] == True:
        print("endx is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["endy_pos"]].isnull().all().values[0] == True:
        print("endy is all empty and cannot be interpolated")
        stop_function = "yes"

    '''NOW SCALING X AND Y ARGS BY Y_SCALE; OK NOW SINCE X AND Y SCALE ARE THE SAME'''
    OCT_pos_pixel = (OCT_pos_mm / y_scale).astype("int32")
    # include lateriality
    OCT_pos_pixel["Laterality"] = im_pd["Laterality"]
    OCT_pos_pixel["series_id"] = im_pd["series"]
    OCT_pos_pixel["Image_aq_time"] = im_pd.Image_aq_time

    # Set all values greater than 768 (grid dim) to 768
    num_pix = OCT_pos_pixel._get_numeric_data()
    num_pix[num_pix > 768] = 767

    return(OCT_pos_pixel, grid,im_pd, stop_function)

def get_xml_indices(Laterality, OCT_pos_pixel, fundus_time, im_pd):
    '''
    :param Laterality: String, laterality of interest
    :param OCT_pos_pixel: data frame, containt all positional arguments in pixels
    :return: x_cord, y_cord, x_start, y_start, x_end, y_end, positional arguments
    '''
    #get series is of the fundus image
    fundus_time_series = im_pd[im_pd.Image_aq_time==fundus_time]["series"]
    # create oct image paths
    y_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "starty_pos", "endy_pos"]]
    # set indices for later positional args
    y_indices.index = range(1, y_indices.shape[0] + 1)
    x_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "startx_pos", "endx_pos"]]
    # interpolate the indices that are missing
    x_indices = x_indices.replace(0, np.nan)
    x_indices = x_indices.interpolate(limit_direction='both')
    # if all values are missing such that not interpolation has been performed, then set back to 0
    x_indices = x_indices.fillna(0)

    # test which of x or y indices is iterated over
    counts = np.unique(y_indices["starty_pos"], return_counts=True)
    if np.max(counts[1]) > 4:
        x_cord = "iterable"
        y_cord = "not_iterable"
    else:
        y_cord = "iterable"
        x_cord = "not_iterable"

    # get integer value of start and end position of OCT scan (apply mode to get the most common one)
    # filters outliers
    x_start = mode(x_indices["startx_pos"])[0][0]
    x_end = mode(x_indices["endx_pos"])[0][0]

    # get integer value of start and end position of OCT scan
    y_end = mode(y_indices["starty_pos"].values)[0][0]
    y_start = mode(y_indices["endy_pos"])[0][0]
    return(x_cord, y_cord, x_start, y_start, x_end, y_end, y_indices, x_indices)