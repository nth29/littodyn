#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from osgeo import osr, gdal, ogr
import sys
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.spatial import distance
import pylab

class Change_detector(object):
    """
    Generic class for change detector
    Implement common methods (read, write, ROI...) to all detectors
    """

    def __init__(self, path_img1,
                 path_img2,
                 path_roi):
        self.path_img1=path_img1
        self.path_img2=path_img2
        self.path_roi=path_roi
        self._load_inputs()


    def _load_inputs(self):
        ds=gdal.Open(self.path_img1, gdal.GA_ReadOnly)
        self.geo=ds.GetGeoTransform()
        self.proj=ds.GetProjection()
        self.cols=ds.RasterXSize
        self.rows=ds.RasterYSize
        self.bands=ds.RasterCount
        self.img1=np.zeros([self.rows,self.cols,self.bands],dtype=np.float)
        for i in range(self.bands):
           self.img1[:,:,i]=ds.GetRasterBand(i+1).ReadAsArray()
        self.img2=np.zeros([self.rows,self.cols,self.bands],dtype=np.float)
        ds=gdal.Open(self.path_img2, gdal.GA_ReadOnly)
        for i in range(self.bands):
           self.img2[:,:,i]=ds.GetRasterBand(i+1).ReadAsArray()
        self._init_mask()

    def _init_mask(self):
        """
        Rasterize roi on dataset area
        """
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource =   driver.Open(self.path_roi, 1)
        layer = dataSource.GetLayer()     
        #Memory dataset for rasterized roi
        target_ds = gdal.GetDriverByName('MEM').Create('', self.cols, self.rows, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(self.geo)
        target_ds.SetProjection(self.proj)
        datainit=np.zeros([self.rows,self.cols],dtype=np.int8)
        myband=target_ds.GetRasterBand(1)
        myband.WriteArray(datainit)
        gdal.RasterizeLayer(target_ds, (1,), layer,burn_values=(1,))
        self.roi_mask=target_ds.GetRasterBand(1).ReadAsArray()

    def detect(self):
        self._dodetect()
        self._apply_roi()

    def _dodetect(self):
        #Generic case here, return 0
        self.change=np.zeros([self.rows,self.cols],dtype=np.float)

    def _apply_roi(self):
        self.change[np.where(self.roi_mask==0)]=np.nan
        
    def save(self, path_out):
        driver = gdal.GetDriverByName('GTiff')
        self.ds = driver.Create(path_out, self.cols, self.rows, 1, gdal.GDT_Float32 )
        self.ds.SetProjection(self.proj)
        self.ds.SetGeoTransform(self.geo)
        self.ds.GetRasterBand(1).WriteArray(self.change)
            


class Change_detector_PCA(Change_detector):
    """
    A change detector with PCA + kmeans
    """

    def _find_vector_set(self, diff_image):
        self.isdata=np.where(self.roi_mask)
        vector_set=np.zeros([len(self.isdata[0]),diff_image.shape[2]],dtype=np.float)
        for i in range(len(self.isdata[0])):
            vector_set[i,:]=diff_image[self.isdata[0][i], self.isdata[1][i],:]
        mean_vec   = np.mean(vector_set, axis = 0)
        vector_set -= mean_vec
    
        return vector_set, mean_vec
    
  
    def _find_FVS(self, EVS, diff_image, mean_vec):
        feature_vector_set=np.zeros([len(self.isdata[0]),diff_image.shape[2]],dtype=np.float)
        for i in range(len(self.isdata[0])):
            feature_vector_set[i,:]=diff_image[self.isdata[0][i], self.isdata[1][i],:]
 
        FVS = np.dot(feature_vector_set, EVS)
        FVS = FVS - mean_vec
        return FVS

    def _clustering(self, FVS, components, new):
    
        kmeans = KMeans(components, verbose = 0)
        kmeans.fit(FVS)
        output = kmeans.predict(FVS)
        count  = Counter(output)

        max_index = max(count, key = count.get)
        change_map=np.zeros(new)+np.nan
        for i in range(len(self.isdata[0])):
            change_map[self.isdata[0][i], self.isdata[1][i]]=output[i]
    
        return max_index, change_map

    def _dodetect(self):
    
        diff_image = np.abs(self.img1 - self.img2)
        
        vector_set, mean_vec = self._find_vector_set(diff_image)
    
        pca     = PCA()
        pca.fit(vector_set)
        EVS = pca.components_
        
        FVS = self._find_FVS(EVS, diff_image, mean_vec)
    
        components = 3
        max_index, self.change = self._clustering(FVS, components, [diff_image.shape[0],diff_image.shape[1]])



class Change_detector_normEUCL(Change_detector):
    """
    A change with only euclidian distance
    """

    def _dodetect(self):
        self.change=np.sqrt(np.sum((self.img1-self.img2)**2,axis=2))

        
class Change_detector_normCORR(Change_detector):
    """
    A change with only correlation distance
    """

    def _dodetect(self):
        dist = np.zeros((self.img1.shape[0],self.img1.shape[1]))
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                dist[i,j] = distance.correlation(self.img1[i,j,:], self.img2[i,j,:])
        self.change=dist

class Change_detector_normCOS(Change_detector):
    """
    A change with only cosine distance
    """

    def _dodetect(self):
        dist = np.zeros((self.img1.shape[0],self.img1.shape[1]))
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                dist[i,j] = distance.cosine(self.img1[i,j,:], self.img2[i,j,:])
        self.change=dist


class Change_detector_VI(Change_detector):
    """
    Generic class for vegetation index based change detectors
    Expected input images bands are
    1 : Blue
    2 : Green
    3 : Red
    4 : Infra Red
    """

    def _dodetect(self):
        """
        For vegetaion indexes we always return abs diff between the vegetation indexes of input images
        """

        vi1=self._vi(self.img1)
        vi2=self._vi(self.img2)
        dist = np.zeros((self.img1.shape[0],self.img1.shape[1]))
        self.change=np.abs(vi2-vi1)

    def _vi(self):
        """
        Vegetation index
        """
        pass

        
class Change_detector_NDVI(Change_detector_VI):
    """
    A change with ndvi
    Normalized vegetation index (https://www.indexdatabase.de/db/i-single.php?id=58)
    Simple
    NDVI = (NIR-R)/(NIR+R)
    """  

    def _vi(self,img):
        """
        ndvi of image
        """
        return (img[:,:,3]-img[:,:,2])/(img[:,:,3]+img[:,:,2])


class Change_detector_EVI(Change_detector_VI):
    """
    A change with evi
    * EVI : enhanced vegetation index (https://www.indexdatabase.de/db/i-single.php?id=16)
    More robuste to atmo effects
    EVI = 2.5 * ((NIR - R) / (NIR + 6 * R – 7.5 * B + 1)) 
    """
 
    def _vi(self,img):
        """
        evi of image
        """
        return 2.5*((img[:,:,3]-img[:,:,2])/(img[:,:,3]+6*img[:,:,2]-7.5*img[:,:,0]+1))


class Change_detector_NGRDI(Change_detector_VI):
    """
    A change with NGRDI
    * NGDRI : https://www.indexdatabase.de/db/i-single.php?id=390)
    For sea bottom
    NGRDI= (Green - Red)/(Green + Red) `
    """

    def _vi(self,img):
        """
        ngrdi of image
        """
        return (img[:,:,1]-img[:,:,2])/(img[:,:,1]+img[:,:,2])



    

if __name__ == "__main__":

    # Command line parser
    parser = argparse.ArgumentParser(
        description="Compute change between images. Example : python change_detector.py ./test_data/20170706_102911_0f43_AnalyticMS_SR.tif ./test_data/20180723_104602_103c_AnalyticMS_SR.tif ./test_data/roi.shp ./test_data/change.tif"

    )
    parser.add_argument(
        "path_img1", type=str, help="Path of input images 1",
    )
    parser.add_argument(
        "path_img2", type=str, help="Path of input images 2",
    )
    parser.add_argument(
        "path_roi", type=str, help="Path of input ROI shapefile",
    )   
    parser.add_argument(
        "path_out", type=str, help="Path of output change image"
    )
    
    args = parser.parse_args()


    dict_algos={"PCA":Change_detector_PCA,
                "EUCL":Change_detector_normEUCL,
                "CORR":Change_detector_normCORR,
                "COS":Change_detector_normCOS,
                "NDVI":Change_detector_NDVI,
                "EVI":Change_detector_EVI,
                "NGRDI":Change_detector_NGRDI}
                

    for mykey in dict_algos.keys():
        myobj=dict_algos[mykey](args.path_img1,
                                args.path_img2,
                                args.path_roi,
        )
        myobj.detect()
        myobj.save(os.path.splitext(args.path_out)[0]+"_"+mykey+".tif")
    
#python change_detector.py ./test_data/spartine_19732595141368_20160504.tif ./test_data/spartine_17918458994113_20180628.tif ./test_data/spartine_roi.shp ./test_data/spartine_change.tif
#python change_detector.py ./test_data/herbiers_19333594309023_20160504.tif ./test_data/herbiers_17383394389091_20181009.tif ./test_data/herbiers_roi.shp ./test_data/herbiers_change.tif
#python change_detector.py ./test_data/ouessant_20170706_102911_0f43_AnalyticMS_SR.tif ./test_data/ouessant_20180707_104507_1014_AnalyticMS_SR.tif ./test_data/ouessant_roi.shp ./test_data/ouessant_change.tif

