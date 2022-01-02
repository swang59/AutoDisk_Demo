#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDisk version 1.0

@author: Sihan Wang（swang59@ncsu.edu）
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import copy
from skimage import feature,draw
from skimage.feature import blob_log
from skimage.io import imsave
from skimage.transform import resize
from scipy import stats,signal



###################################################################
# This file includes the utilities of AutoDisk, an automated diffraction 
# pattern analysis method for 4D-STEM. This version covers the functions
# for diffraction disk recognition, lattice parameter estimation and
# lattice strain mapping.
#
# For details about the method, please refer to the manuscript:
# "AutoDisk: Automated Diffraction Processing and Strain Mapping in 4D-STEM"
# by Sihan Wang, Tim Eldred, Jacob Smith and Wenpei Gao.
###################################################################


def visual(image,plot = True):
    """
    Convert a 2D array of int or float to an int8 array of image and visualize it.

    Parameters
    ----------
    image : 2D array of int or float
    plot : bool, optional
        Ture if the image need to be ploted. The default is True.

    Returns
    -------
    image_out: 2D array of int8

    """
    image_out = (((image - image.min()) / (image.max() - image.min())) * 255).astype(np.uint8)
    if plot==True:
        plt.imshow(image_out,cmap='gray')
        plt.show()
        
    return image_out



def readData(dname):   
    """
    Read in a 4D-STEM data file.
    
    Parameters
    ----------
    dname : str
        Name of the data file.

    Returns
    -------
    data: 4D array of int or float
        The read-in 4D-STEM data.
    
    """
    dimy = 130
    dimx = 128

    file = open(dname,'rb') 
    data = np.fromfile(file, np.float32)          
    pro_dim = int(np.sqrt(len(data)/dimx/dimy))
    
    data = np.reshape(data, (pro_dim, pro_dim, dimy, dimx))
    data = data[:,:,1:dimx+1, :]
    file.close()
    
    return data    
        
        
        
def savePat(out_dir, data, ext ='.tif'):
    """
    Save diffraction patterns into '.tif's.

    Parameters
    ----------
    out_dir : str
        The name of the save folder.
    data : 2D array of int or float
        Array of a 4D dataset.
    ext : str, optional
        Extension of the output pattern. The default is '.tif'.

    Returns
    -------
    None.

    """
    pro_dim,pro_dim = data.shape[:2]
    out_dir = os.path.join(out_dir)
    for i in range(pro_dim):
        for j in range(pro_dim):
            pattern = data[i,j]       
            imsave(out_dir+np.str_(i)+'_'+np.str_(j)+".tif", pattern, plugin="tifffile")
    
    pass



def generateAdf(data,in_rad,out_rad,save=False): 
    """
    Generate an annular dark-field image from the diffraction patterns.

    Parameters
    ----------
    data : 4D array of int or float
        The 4D dataset.
    in_rad : int
        Inner collection angle.
    out_rad : int
        Outer collection angle.

    Returns
    -------
    None.

    """
    imgh,imgw,pxh,pxw = data.shape
    i = imgh//2
    j = imgw//2
    
    data[np.where(np.isnan(data)==True)] = 0
    data[i,j,:,:] -= np.min(data[i,j,:,:])
    data[i,j,:,:] += 0.0000000001
    
    mask_img = np.zeros((pxh,pxw,3))

    rr,cc = draw.disk((pxh//2,pxw//2),out_rad)
    draw.set_color(mask_img,[rr,cc],[1,1,1])

    rr,cc = draw.disk((pxh//2,pxw//2),in_rad)
    draw.set_color(mask_img,[rr,cc],[0,0,0])

    adf=np.mean(data*mask_img[:,:,0],axis=(-2,-1))

    plt.imshow(adf,cmap='gray')
    plt.show() 

    pass 
  


def generateAvg(data):
    """
    Generate an average (sum) pattern from the 4D dataset.

    Parameters
    ----------
    data : 4D array of int or float
        Array of the 4D dataset.

    Returns
    -------
    avg_pat: 2D array of int or float
        An average (sum) difffraction pattern.

    """
    pro_y,pro_x = data.shape[:2]
    avg_pat = data[0,0]*1
    avg_pat[:,:] = 0
    for row in range (pro_y):
        for col in range (pro_x):
            avg_pat += data[row,col]
    
    return avg_pat

          

def ctrRadiusIni(pattern):
    """
    Find the center coordinate and the radius of the zero-order disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.

    Returns
    -------
    ctr : 1D array of int or float
        Array of the center coordinates [row,col].
    avg_r : float
        Radius of the center disk in unit of pixels.

    """
    h,w = pattern.shape
    ctr = h//2
    pix_w = pattern[ctr,:]
    pix_h = pattern[:,ctr]
    
    fir_der_w = np.abs(pix_w[:1]-pix_w[1:])
    sec_dir_w_r = np.array(fir_der_w[w//2:-1]-fir_der_w[w//2+1:])
    sec_dir_w_l = np.array(fir_der_w[1:w//2]-fir_der_w[:w//2-1])
    avg_pos1_w = np.where(sec_dir_w_r==sec_dir_w_r.max())[0][0]
    avg_pos2_w = np.where(sec_dir_w_l==sec_dir_w_l.max())[0][0]
    avg_r_w = np.mean([avg_pos1_w+1,len(sec_dir_w_l)-avg_pos2_w])
    ctr_w =  np.mean([w//2 + avg_pos1_w + 1,avg_pos2_w + 2])
    
    fir_der_h = np.abs(pix_h[:1]-pix_h[1:])
    sec_dir_h_b = np.array(fir_der_h[h//2:-1]-fir_der_h[h//2+1:])
    sec_dir_h_u = np.array(fir_der_h[1:h//2]-fir_der_h[:h//2-1])
    avg_pos1_h = np.where(sec_dir_h_b==sec_dir_h_b.max())[0][0]
    avg_pos2_h = np.where(sec_dir_h_u==sec_dir_h_u.max())[0][0]
    avg_r_h = np.mean([avg_pos1_h+1,len(sec_dir_h_u)-avg_pos2_h])
    ctr_h =  np.mean([h//2 + avg_pos1_h + 1, avg_pos2_h+2])
    
    avg_r = np.mean([avg_r_w,avg_r_h])
    ctr = np.array([ctr_h,ctr_w])
    
    return ctr,avg_r            
            


def genrateKernel(pattern,ctr,r,c=0.7,pad=2,pre_def = False):
    """
    Generate the kernel for cross-correlation based on thee center disk.

    Parameters
    ----------
    pattern : 2D array of int or float
        An array of a diffraction pattern.
    ctr : 1D array of float
        Array of the row and column coordinates of the center.
    r : float
        Radius of a disk.
    c : float, optional
        An coefficient to modify the kernel size. The default is 0.7.
    pad : int, optional
        A hyperparameter to change the padding size out of the feature. The default is 2.
    pre_def: bool, optional
        If True, read the pre-defined ring kernel. The default is False.

    Returns
    -------
    fil_ring : 2D array of float
        Array of the kernel.

    """
    if pre_def == True:
        ring = np.load("kernel_cir.npy")
        f_size = int(2*r*c)
        ring = resize(ring, (f_size, f_size))
        fil_ring = np.zeros((len(ring)+2*pad,len(ring)+2*pad),dtype=float)
        fil_ring[pad:-pad,pad:-pad] = ring

        return fil_ring
    
    
    y_st = int(ctr[0]-r+0.5-pad*2)
    y_end = int(ctr[0]+r+0.5+pad*2)
    x_st = int(ctr[1]-r+0.5-pad*2)
    x_end = int(ctr[1]+r+0.5+pad*2)
    # +0.5 to avoid rounding errors (always shift to right, so 0,5 is modified to 1.5)
    
    if y_end-y_st==x_end-x_st:
        ctr_disk = pattern[y_st:y_end,x_st:x_end] 
    elif y_end-y_st>x_end-x_st:
        ctr_disk = pattern[y_st+1:y_end,x_st:x_end] 
    else:
        ctr_disk = pattern[y_st:y_end,x_st+1:x_end] 
        
    edge_det = feature.canny(ctr_disk, sigma=1)
    
    dim = len(ctr_disk)
    dim_hf = dim/2
    fil_ring = np.zeros((dim,dim))
    for i in range (dim):
        for j in range (dim):
            if edge_det[i,j]==True:
                if (i-dim_hf)**2+(j-dim_hf)**2>int(r-2)**2 and (i-dim_hf)**2+(j-dim_hf)**2<int(r+2)**2:
                    fil_ring[i,j] = 1
    
    coef = int(c*r)
    f_size = 2*coef
    fil_ring = resize(fil_ring, (f_size, f_size))
    
    return fil_ring



def crossCorr(pattern,kernel):
    """
    Cross correlate the pattern with the kernal.

    Parameters
    ----------
    pattern : 2D array of int or float
        Array of a diffraction pattern to be cross correlated.
    kernel : 2D array of float
        Array of the kernel.

    Returns
    -------
    cro_img_out : 2D array
        Cross correlated result of the input pattern.

    """
    cro_cor_img = signal.correlate2d(pattern, kernel, boundary='symm', mode='same')
    cro_img_out = np.sqrt(cro_cor_img)
    

    return cro_img_out



def samePadding(img,kernel):
    """
    Generate a padding outside of the image with the average intensity on the boundary of the image.

    Parameters
    ----------
    img : 2D array of int or float
        Array of the image.
    kernel : 2D array of float
        Array of the kernel.

    Returns
    -------
    constant : 2D array
        The image with a constant padding.
        
    """
    f_size = len(kernel)
    constant = np.empty((img.shape[0]+2*f_size,img.shape[1]+2*f_size))
    bcgd = np.mean(img[:f_size,f_size:])
    constant[0:f_size,:] = constant[-f_size:img.shape[0]+2*f_size,:] = constant[:,0:f_size] = constant[:,f_size:img.shape[1]+2*f_size] = bcgd
    constant[f_size:img.shape[0]+f_size,f_size:img.shape[1]+f_size] = img
    
    return constant



def ctrDet(pattern, r, kernel, n_sigma=10, thred=0.1, ovl=0):
    """
    Detect disks on a pattern.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    r : float
        Radius of a disk.
    kernel : 2D array of float
        Kernel used for cross correlation.
    n_sigma : int, optional
        The number of intermediate values of standard deviations. The default is 10.
    thred : float, optional
        The absolute lower bound for scale space maxima. The default is 0.1.
    ovl : float, optional
        Acceptable overlapping area of the blobs. The default is 0.

    Returns
    -------
    blobs : 2D array of int
         Corrdinates of the detected disk position.

    """
    adjr = r * 0.5
    
    img = samePadding(pattern,kernel)  
    sh,sw = img.shape

    blobs_log = blob_log(img, 
                 min_sigma=adjr,
                 max_sigma=adjr, 
                 num_sigma=n_sigma, 
                 threshold= thred,
                 overlap = ovl)    
    
    rem = []
    f_size = len(kernel)
    for i in range (len(blobs_log)):
        if np.any(blobs_log[i,:2]<f_size+5) or np.any(blobs_log[i,0]>sh-f_size-5) or np.any(blobs_log[i,1]>sw-f_size-5):
            rem.append(i)
    
    blobs_log_out = np.delete(blobs_log, rem, axis =0)
    blobs_log_out -= f_size 
    
    blobs =  blobs_log_out[:,:2].astype(int)
    
    return blobs



def radGradMax(sample, blobs, r, rn=20, ra=2, n_p=40, threshold=3): 
    """
    Radial gradient Maximum process.

    Parameters
    ----------
    sample : 2D array of float or int
        The diffraction pattern.
    blobs : 2D array of int or float
        Blob coordinates.
    r : float
        Radius of the disk
    rn : int, optional
        The total number of rings. The default is 20.
    ra : int, optional
        Half of the window size. The default is 2.
    n_p : int, optional
        The number of sampling points on a ring. The default is 40.
    threshold : float, optional
        A threshold to filter out outliers. The smaller the threshold is, the more outliers are detected. The default is 3.

    Returns
    -------
    ref_ctr : 2D array of float
        Array with three columns, y component, x component and the weight of each detected disk.

    """
    ori_ctr = blobs    
    h,w = sample.shape        
    adjr = r * 1   
    r_scale = np.linspace(adjr*0.8, adjr*1.2, rn)    
    theta = np.linspace(0, 2*np.pi, n_p)     
    ref_ctr = []

    for lp in range (len(ori_ctr)):
        test_ctr = ori_ctr[lp]
        ind_list = []
        for ca in range (-ra,ra):
            for cb in range (-ra,ra):
                cur_row, cur_col = test_ctr[0]+ca, test_ctr[1]+cb
                cacb_rn = np.empty(rn)
                for i in range (rn):
                    row_coor = np.array([cur_row + r_scale[i] * np.sin(theta) + 0.5]).astype(int)
                    col_coor = np.array([cur_col + r_scale[i] * np.cos(theta) + 0.5]).astype(int)
                    
                    row_coor[row_coor>=h]=h-1
                    row_coor[row_coor<0]=0
                    col_coor[col_coor>=w]=w-1
                    col_coor[col_coor<0]=0
                    
                    int_sum = np.sum(sample[row_coor,col_coor])
                    cacb_rn[i] = int_sum
                    
                cacb_rn[:rn//2] *= np.linspace(1,rn//2,rn//2) 
                cacb_diff = np.sum(cacb_rn[:rn//2]) - np.sum(cacb_rn[rn//2:])
                ind_list.append([cur_row, cur_col,cacb_diff])
                
        
        ind_list = np.array(ind_list) 
        ind_max = np.where(ind_list[:,2]==ind_list[:,2].max())[0][0]
        ref_ctr.append(ind_list[ind_max]) 

    ref_ctr = np.array(ref_ctr)

    # Check Outliers
    z = np.abs(stats.zscore(ref_ctr[:,2]))
    outlier = np.where(z>threshold)
    if len(outlier[0])>0:
        for each in outlier[0]:
            if np.linalg.norm(ref_ctr[each,:2]-[h//2,w//2])> r:
                ref_ctr = np.delete(ref_ctr,outlier[0],axis = 0)

    return ref_ctr



def detAng(ref_ctr,ctr,r): # threshold: accepted angle difference
    """
    Detect an angle to rotate the disk coordinates.

    Parameters
    ----------
    ref_ctr : 2D array of float
        Array of disk position coordinates and their corresponding weights
    ctr : 1D array of float
        Center of the zero-order disk.
    r : float
        Radius of the disks.

    Returns
    -------
    wt_ang : float
        The rotation angle.
    ref_ctr : 2D array of float
        Refined disk positions.

    """
    ctr_vec = ref_ctr[:,:2] - ctr
    ctr_diff = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
    ctr_idx = np.where(ctr_diff==ctr_diff.min())[0][0]
    
    diff = ref_ctr[:,:2]-ctr
    distance = diff[:,0]**2 + diff[:,1]**2
    
    dis_copy = copy.deepcopy(distance)
    min_dis = []
    while len(min_dis) <5:
        cur_min = dis_copy.min()
        idx_rem = np.where(dis_copy==cur_min)[0]
        dis_copy = np.delete(dis_copy,idx_rem)
        idx_ctr = np.where(distance==cur_min)[0]
        if len(idx_ctr)==1:
            min_dis.append(ref_ctr[idx_ctr[0],:2])
        else:
            for each in idx_ctr:   
                min_dis.append(ref_ctr[each,:2])

    min_dis_ctr = np.array(min_dis,dtype = int)
    min_dis_ctr = np.delete(min_dis_ctr,0,axis = 0) # delete [0,0]

    vec = min_dis_ctr-ctr
       
    ang = np.arctan2(vec[:,0],vec[:,1])* 180 / np.pi
    
    for i in range (len(ang)):
        ang[i] = (180 + ang[i]) if (ang[i]<0) else ang[i]

    cand_ang_idx = np.where(ang==ang.min())[0]
    sup_pt = min_dis_ctr[cand_ang_idx] # the point retuning the smallest rotation angle


    ref_diff = ctr-sup_pt
    ini_ang = np.arctan2(ref_diff[:,0],ref_diff[:,1])*180/np.pi
    all_ref = []
    for n in range (len(ini_ang)):
        all_ref.append(np.array([ini_ang[n]]))
    if len(ref_diff)>1:
        ref_diff = ref_diff[0]

    for each_ctr in ref_ctr:
        cur_vec = each_ctr[:2] - ref_diff
        cur_diff = ref_ctr[:,:2]-cur_vec
        cur_norm = np.linalg.norm(cur_diff,axis=1)
        if cur_norm.min()<r:
            ref_idx = np.where(cur_norm==cur_norm.min())[0]
            ref_pt = ref_ctr[ref_idx]
            ref_vec = ref_pt - each_ctr
            all_ref.append(np.arctan2(ref_vec[:,0],ref_vec[:,1])* 180 / np.pi)
    
    for i in range (len(all_ref)):
        if all_ref[i]<0:
            all_ref[i] = 180 + all_ref[i]
        elif all_ref[i] >= 180:
            all_ref[i] = 180 - all_ref[i]
        
    wt_ang = np.mean(all_ref)
    ref_ctr[ctr_idx,2] = 10**38
    
    return wt_ang, ref_ctr



def rotImg(image, angle, ctr):
    """
    Rotate a pattern.

    Parameters
    ----------
    image : 2D array of int or float
        The input pattern.
    angle : float
        An angel to rotate.
    ctr : 1D array of int or float
        The rotation center.

    Returns
    -------
    result : 2D array of int or float
        The rotated pattern.

    """
    image_center = tuple(np.array([ctr[0],ctr[1]]))
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    return result



def rotCtr(pattern,ref_ctr,angle):
    """
    Rotate disk coordinates.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ref_ctr : 2D array of float
        Array of the detected disk positions.
    angle : float
        Detected angle to rotate.

    Returns
    -------
    ctr_new : 2D array of float
        The transformed disk positions.

    """
    h,w = pattern.shape
    ctr_idx = np.where(ref_ctr[:,2]==ref_ctr[:,2].max())[0][0]
    ctr = ref_ctr[ctr_idx]
    ctr_new = []
    ang_rad = angle*np.pi/180

    for i in range (len(ref_ctr)):
        cur_cd = ref_ctr[i,:2]
        y_new = -(ctr[0] - (cur_cd[0]-ctr[0])*np.cos(ang_rad) + (cur_cd[1]-ctr[1])*np.sin(ang_rad) ) + 2*ctr[0]
        x_new = (ctr[1] + (cur_cd[0]-ctr[0])*np.sin(ang_rad) + (cur_cd[1]-ctr[1])*np.cos(ang_rad) )
        
        if y_new>0 and x_new>0 and y_new<h and x_new<w:
            ctr_new.append([y_new,x_new,ref_ctr[i,2]])
    
    ctr_new = np.array(ctr_new)    

    return ctr_new



def groupY (load_ctr,r):
    """
    Group disks based on their row coordinates.

    Parameters
    ----------
    load_ctr : 2D array of float
        Array of disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    g_y : a list of arrays of float
        A list with each element as a group of disk positions.

    """
    n = len(load_ctr)
    
    g_y = [[load_ctr[0,:]]]
    for i in range (1,n):        
        gy_mean = []
        for group in g_y:
            cur_mean = 0
            grp_len = len(group)
            for each in group:
                cur_mean += each[0]
            apd_mean = cur_mean/grp_len
            gy_mean.append(apd_mean)
        
        diffy = [np.abs(s-load_ctr[i,0]) for s in gy_mean]
        gy_ind = np.argmin(diffy) 
        min_diffy = np.min(diffy)
        if min_diffy>r:
            g_y.append([load_ctr[i]])
        else:
            g_y[gy_ind].append(load_ctr[i])

    return g_y



def latFit(pattern,rot_ref_ctr,r):  
    """
    Lattice fitting process.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    rot_ref_ctr : 2D array of float
        Array of the disks positionss.
    r : float
        Radius of the disks.

    Returns
    -------
    vec_a : 1D array of float
        The estimated horizontal lattice vector [y component, x component].
    vec_b_ref : 1D array of float
        The estimated non-horizontal lattice vector [y component, x component].
    result_ctr : 2D array of float
        Array of the refined disk positions.
    lat_ctr_arr : 2D array of float
        The array of the positions of disks in the middle row.
    avg_ref_ang : float
        Refined rotation angle.

    """ 
    load_ctr = rot_ref_ctr*1
    g_y = groupY(load_ctr,r)
    
    vec_a = np.array([0,0])
    vec_b_ref = np.array([0,0])
    
    result_ctr = copy.deepcopy(rot_ref_ctr)
    lat_ctr = []
    avg_ref_ang = 0
    
    ########## Sort y values in each group and refine the angle ##########
    ref_ang = []
    for ea_g in g_y:
        if len(ea_g)>1:
            ea_g_arr = np.array(ea_g)
        
            result = np.polyfit(ea_g_arr[:,1], ea_g_arr[:,0], 1)
            ref_ang.append(np.arctan2(result[0],1)* 180 / np.pi)
    
    if len(ref_ang)>0:
        avg_ref_ang =  sum(ref_ang)/len(ref_ang) 
    else:
        avg_ref_ang = 0
        
    rot_ref_ctr2 = rotCtr(pattern,load_ctr,avg_ref_ang)
    
    g_y = groupY(rot_ref_ctr2,r)

    g_y_len = [len(l) for l in g_y]
    
    if max(g_y_len)>1:
        ################ Refine y values #######################            
        n = len(rot_ref_ctr2)
        ref_y = []
        for group in g_y:
            cur_mean = 0
            sum_cur = 0
            for each in group:
                sum_cur += each[2]
            for each in group:
                cur_mean += each[0]*(each[2]/sum_cur)
            ref_y.append(cur_mean) # Weighted mean     
            
        # Change y values to the averaged y in each group    
        result_ctr = copy.deepcopy(rot_ref_ctr2)
        for j in range (n):
            cur_y = rot_ref_ctr2[j,0]
            d_y = [np.abs(s-cur_y) for s in ref_y]
            min_y_ind = np.argmin(d_y)
            result_ctr[j][0] = ref_y[min_y_ind]     
        
        ################ Vec a #######################    
        x_g = []    
        tit_diff_x = []  
        for cur_y in ref_y:
            cur_x_g = result_ctr[np.where(result_ctr[:,0]== cur_y)]
            if len(cur_x_g)>1:
                cur_x_g.sort(axis = 0)
                x_g.append(cur_x_g)
                cur_diff_x = cur_x_g[1:]-cur_x_g[:-1]
                tit_diff_x.append(cur_diff_x)
            else:
                x_g.append(cur_x_g)   
        
        ###################### Calculate average distance ################
        if len(tit_diff_x)>0:
            outl_rem_x = []
            mean_diff_x = []
            
            for i in range (len(tit_diff_x)):
                for x in tit_diff_x[i]:
                    outl_rem_x.append(x[1])
                    
            outl_rem_x = np.array(outl_rem_x)
            q1, q3= np.percentile(outl_rem_x,[25,75])
            lower_bound = 2.5*q1 - 1.5*q3
            upper_bound = 2.5*q3 - 1.5*q1
            
            for each_g in tit_diff_x:
                each_g_mod = each_g*1
                for idx in range (len(each_g)):
                    if each_g[idx,1]<lower_bound or each_g[idx,1]>upper_bound:
                        each_g_mod = np.delete(each_g,idx,axis = 0)               
                
                if len(each_g_mod)>0:
                    cur_mean = np.mean(each_g_mod[:,1],axis=0)
                    mean_diff_x.append([cur_mean,len(each_g_mod)])
                
            mean_diff_x_arr = np.array(mean_diff_x)
            
            if len(mean_diff_x_arr)>0:
                count = 0 
                sum_x = 0
                for i in range (len(mean_diff_x_arr)):
                    sum_x += mean_diff_x_arr[i,0]* mean_diff_x_arr[i,1]
                    count += mean_diff_x_arr[i,1]
                
                vec_a = np.array([0, sum_x/count])
                
                ######### Find vector b #########
                set_ct_ind = np.argmax(result_ctr[:,2])
                set_ct = result_ctr[set_ct_ind]
                
                # Find rough b
                min_nn = 10**38
                nn_vecb_rough = np.array([-1,-1,-1])
                for gn in range (len(x_g)):
                    cur_ct = x_g[gn]
                    if set_ct[0] not in cur_ct[:,0]:
                        dis_xy = cur_ct - set_ct
                        dis_norm = np.linalg.norm(dis_xy[:,:2],axis = 1)
                        xy_min = np.min(dis_norm)
                        if xy_min<=min_nn:  
                            min_nn = xy_min 
                            nn_vecb_rough = cur_ct[np.argmin(dis_norm)]   
                
                # Generate hypothetical lattice
                h,w = pattern.shape 
                lat_ctr = [set_ct[:2]]
                
                ###### Generate pts along vector a (middle row) ######
                # one side    
                cur_h1 = set_ct[0]
                cur_w1 = set_ct[1]
                cur_ct1 = set_ct[:2]*1
                while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                        cur_h1,cur_w1 = cur_ct1-vec_a
                        if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                            cur_ct1 = [cur_h1,cur_w1]
                            lat_ctr.append([cur_h1,cur_w1])
                
                # the other side
                cur_h2 = set_ct[0]
                cur_w2 = set_ct[1]
                cur_ct2 = set_ct[:2]*1.0
                while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                    cur_h2,cur_w2 = cur_ct2+vec_a
                    if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                        cur_ct2 = [cur_h2,cur_w2]
                        lat_ctr.append([cur_h2,cur_w2])                            
                        
                ######### Refine Vector b #########
                vec_b = nn_vecb_rough - set_ct
                if  vec_b[0]<0:
                    vec_b = -vec_b
            
                vec_b_rough = vec_b [:2]
            
                diff_y_ref = []   

                look_y = set_ct[0]-vec_b_rough[0]
                est_ct = lat_ctr - vec_b_rough 
                while look_y>0:
                    for each in est_ct:
                        each_diff_xy = each - result_ctr[:,:2]
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)
                    look_y -= vec_b_rough[0]
                    est_ct -= vec_b_rough
        
                look_y = set_ct[0]+vec_b_rough[0]
                est_ct = lat_ctr + vec_b_rough
                while look_y<h:
                    for each in est_ct:
                        each_diff_xy = result_ctr[:,:2] - each
                        
                        each_dis = each_diff_xy[:,0]**2+each_diff_xy[:,1]**2
                        each_dis_min = np.min(each_dis)
        
                        if each_dis_min<r**2:
                            cum_row = round(np.abs(np.mean(each[:][0])-set_ct[0])/vec_b_rough[0])
                            diff_y_ref.append(each_diff_xy[np.argmin(each_dis)]/cum_row)   
                    look_y += vec_b_rough[0]
                    est_ct += vec_b_rough      
                 
                vec_b_ref = vec_b_rough*1.0
                if len(diff_y_ref)==0:
                    diff_y_ref.append([0,0])
                diff_y_ref = np.array(diff_y_ref)        
                vec_b_ref[1] = vec_b_ref[1] + np.mean(diff_y_ref[:,1])
    
    lat_ctr_arr = np.array(lat_ctr)
    return vec_a, vec_b_ref, result_ctr, lat_ctr_arr, avg_ref_ang



# Generate 2d lattice based on vector a and b
def genLat(pattern, ret_a,ret_b, mid_ctr,r):
    """
    Generate a matrix of hypothetical lattice points.

    Parameters
    ----------
    pattern : 2D array of int or float
        A diffraction pattern.
    ret_a : 1D array of float
        The horizontal lattice vector a.
    ret_b : 1D array of float
        The non-horizontal lattice vector b.
    mid_ctr : a list of arrays of float
        a list of disk positions which are in the middle row.
    r : float
        Radius of the disks.

    Returns
    -------
    final_ctr : 2D array of float
        Disk positions in the hypothetical lattice.

    """
    img = pattern
    veca,vecb = ret_a,ret_b
    h,w = img.shape
    veca_ct = mid_ctr[:,:2].copy()
    final_ctr = []
    
    for cur_veca_ct in veca_ct:
        # one side    
        cur_h1 = cur_veca_ct[0]
        cur_w1 = cur_veca_ct[1]
        cur_ct1 = cur_veca_ct*1

        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
            cur_h1,cur_w1 = cur_ct1-vecb
            if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_ct1 = [cur_h1,cur_w1]
                final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_veca_ct[0]
        cur_w2 = cur_veca_ct[1]
        cur_ct2 = cur_veca_ct*1

        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+vecb
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]
                final_ctr.append([cur_h2,cur_w2])  

    ########   Check Again ########
    chk_lat_ctr= final_ctr
    
    for cur_vec2_ct in chk_lat_ctr:
        # one side    
        cur_h1 = cur_vec2_ct[0]
        cur_w1 = cur_vec2_ct[1]
        cur_ct1 = cur_vec2_ct*1
        while cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                cur_h1,cur_w1 = cur_ct1-veca
                # print(cur_ct1-veca,cur_h1,cur_w1)
                if cur_h1>=0 and cur_h1<=h and cur_w1>=0 and cur_w1<=w:
                    cur_ct1 = [cur_h1,cur_w1]
                    dif_chk = [(ct[0]-cur_ct1[0])**2+(ct[1]-cur_ct1[1])**2 for ct in chk_lat_ctr]
                    if min(dif_chk)> r**2: 
                        final_ctr.append([cur_h1,cur_w1])
        
        # the other side
        cur_h2 = cur_vec2_ct[0]
        cur_w2 = cur_vec2_ct[1]
        cur_ct2 = cur_vec2_ct*1
        while cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
            cur_h2,cur_w2 = cur_ct2+veca
            if cur_h2>=0 and cur_h2<=h and cur_w2>=0 and cur_w2<=w:
                cur_ct2 = [cur_h2,cur_w2]   
                dif_chk2 = [(ct[0]-cur_ct2[0])**2+(ct[1]-cur_ct2[1])**2 for ct in chk_lat_ctr]
                if min(dif_chk2)> r**2:  
                    final_ctr.append([cur_h2,cur_w2])   
                                 
    for pt in mid_ctr:
        final_ctr.append(pt)
    
    final_ctr = np.array(final_ctr)
                    
    return final_ctr



def delArti(gen_lat_pt,ref_ctr,r):
    """
    Delete any artificial lattice points.

    Parameters
    ----------
    gen_lat_pt : 2D array of float
        Array of artificial disk positions.
    ref_ctr : 2D array of float
        Array of detected disk positions.
    r : float
        Radius of the disks.

    Returns
    -------
    gen_lat_pt_up : 2D array of float
        A filtered array of disk positions.

    """
    gen_lat_pt_up = []
    for i in range (len(gen_lat_pt)):
        dif_gen_ref = np.array(gen_lat_pt[i] - ref_ctr[:,:2])
        dif_gen_ref_norm = np.linalg.norm(dif_gen_ref,axis = 1)
        if dif_gen_ref_norm.min()< r:
            gen_lat_pt_up.append(gen_lat_pt[i])
    
    gen_lat_pt_up = np.array(gen_lat_pt_up)
    
    return gen_lat_pt_up



def latBack(refe_a,refe_b,angle):
    """
    Transform the lattice vectors to the default coordinate system.

    Parameters
    ----------
    refe_a : 1D array of float
        Array of the vector a.
    refe_b : 1D array of float
        Array of the vector b.
    angle : float
        The rotation angle.

    Returns
    -------
    a_init : 1D array of float
        Transformed array of the vector a.
    b_init : 1D array of float
        Transformed array of the vector b.

    """
    ang_init_back = angle*np.pi/180
    a_init = np.array([refe_a[1]*np.sin(ang_init_back),refe_a[1]*np.cos(ang_init_back)])
    b_init = np.array([refe_b[1]*np.sin(ang_init_back)+refe_b[0]*np.cos(ang_init_back),refe_b[1]*np.cos(ang_init_back)-refe_b[0]*np.sin(ang_init_back)])
    
    return a_init,b_init



def drawCircles(ori_pattern,blobs_list,r):
    """
    Label the disk positions on the pattern.

    Parameters
    ----------
    ori_pattern : 2D array of int or float
        The pattern to be labeled on.
    blobs_list : 2D array of float
        Array of disk positions.
    r: float
        The radius of the disks.

    Returns
    -------
    None.

    """
    pattern = copy.deepcopy(ori_pattern)
    
    for q in range (len(blobs_list)):
        center = (int(blobs_list[q,0]),int(blobs_list[q,1]))
        pattern[center] = pattern.max()
    
    fig, ax = plt.subplots(figsize = (5,5))
    ax.imshow(pattern,cmap='gray')
    for blob in blobs_list:
        y, x = blob
        c = plt.Circle((x, y),r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)
    
    plt.show()
    
    pass



def latDist(lat_par,refe_a,refe_b,err=0.2):
    """
    This function filters out the outliers of the lattice parameters based on the references.

    Parameters
    ----------
    lat_par : 2D array of arrays of float
        2D array with each element as two arrays of lattice vectors.
    refe_a : 1D array of float
        The reference lattice vector a.
    refe_b : 1D array of float
        The reference lattice vector b.
    err : float, optional
        Acceptable error percentage. The default is 0.2 (20%).

    Returns
    -------
    store_whole : 3D array of float
        Array containing 3 columns, y coordinate, x coordinate, and 4 lattice vector elements
        (y of vector a, x of vector a, y of vector b, x of vector b).

    """
    arr_vec = lat_par
    
    sm_y,sm_x = lat_par.shape[:2]
    std_ax = refe_a[1] # vec_a[0,std_2x]
    std_ay = refe_a[0]
    std_bx = refe_b[1] # vec_b[std_1y,std_1x]
    std_by = refe_b[0]
    
    acc_ax_min = std_ax*(1-err) if std_ax>0 else std_ax*(1+err)
    acc_ax_max = std_ax*(1+err) if std_ax>0 else std_ax*(1-err)
    acc_ay_min = std_ay*(1-err) if std_ay>0 else std_ay*(1+err)
    acc_ay_max = std_ay*(1+err) if std_ay>0 else std_ay*(1-err)
    acc_bx_min = std_bx*(1-err) if std_bx>0 else std_bx*(1+err)
    acc_bx_max = std_bx*(1+err) if std_bx>0 else std_bx*(1-err)
    acc_by_min = std_by*(1-err) if std_by>0 else std_by*(1+err)
    acc_by_max = std_by*(1+err) if std_by>0 else std_by*(1-err)
    
    store_whole = np.zeros((sm_y,sm_x,4),dtype = float)

    # Delete paramater outliers
    ct = 0
    for row in range (sm_y):
        for col in range (sm_x):
            
            each = arr_vec[row,col]
        
            gax = float(each[0,1])
            gay = float(each[0,0])
            gbx = float(each[1,1])  
            gby = float(each[1,0])
            
            if gax>acc_ax_max or gax<acc_ax_min or gay>acc_ay_max or gay<acc_ay_min or gbx>acc_bx_max or gbx<acc_bx_min or gby>acc_by_max or gby<acc_by_min:
                ct += 1
    
            else:
                store_whole[row,col][0] = gay        
                store_whole[row,col][1] = gax
                store_whole[row,col][2] = gby
                store_whole[row,col][3] = gbx                

    return store_whole       



def calcStrain(lat_fil, refe_a,refe_b):
    """
    Compute strain maps.
    
    Parameters
    ----------
    lat_fil : 2D array of arrays of float
        2D array with each element as two lattice vectors.
    refe_a : 1D array of float 
        The reference vector a.
    refe_b : 1D array of float
        The reference vector b.

    Returns
    -------
    st_xx : 2D array of float
        Estimated strain along the x direction.
    st_yy : 2D array of float
        Estimated strain along the y direction.
    st_xy : 2D array of float
        Shear strain.
    st_yx : 2D array of float
        Shear strain.
    tha_ang : 2D array of float
        Angle of lattice rotation in deg.

    """
    sm_y,sm_x = lat_fil.shape[:2]
    
    st_xx = np.zeros((sm_y,sm_x),dtype=float)
    st_yx = np.zeros((sm_y,sm_x),dtype=float)
    st_xy = np.zeros((sm_y,sm_x),dtype=float)
    st_yy = np.zeros((sm_y,sm_x),dtype=float)
    tha_ang = np.zeros((sm_y,sm_x),dtype=float)
    
    G0_T = np.array([[refe_a[1],refe_a[0]],[refe_b[1],refe_b[0]]])
    
    for row in range (sm_y):
        for col in range (sm_x):
            if any(lat_fil[row,col]!=0):
                gay,gax,gby,gbx = lat_fil[row,col]
    
                G = np.array([[gax,gbx],[gay,gby]])
                G_T = np.transpose(G)
                G_T_n1 = np.linalg.inv(G_T)
                
                D = G_T_n1.dot(G0_T)
                theta = np.arctan2((D[1,0]-D[0,1]),(D[0,0]+D[1,1]))
                
                M = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
                
                F = M.dot(D)
                I = np.array([[1,0],[0,1]])
                
                eps = F-I
                
                st_xx[row,col] = eps[0,0]
                st_yy[row,col] = eps[1,1] 
                st_xy[row,col] = eps[0,1]
                st_yx[row,col] = eps[1,0] 
                tha_ang[row,col] = theta/np.pi*180

    return st_xx,st_yy,st_xy,st_yx,tha_ang


