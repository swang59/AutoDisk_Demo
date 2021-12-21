#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:33:11 2021

@author: Sihan Wang（swang59@ncsu.edu）
"""

from autodisk import *


# Input the file name of 4D-STEM data.
data_name = 'pdpt_x64_y64.raw'

data_ori = readData(data_name)


imgh,imgw,pxh,pxw = data_ori.shape

print('Dimension of the scanning area: {} * {}; Dimension of each diffraction pattern: {} * {}.'.format(imgh,imgw,pxh,pxw))


# savePat('pattern/', data_ori, ext ='.tif')
#

# inner and outer radius
in_rad = 12
out_rad = 50

generateAdf(data_ori,in_rad,out_rad)


data = copy.deepcopy(data_ori)

data[np.where(np.isnan(data)==True)] = 0

data -= data.min() if data.min()< 0 else 0
data += 10**(-17)


avg_pat = generateAvg(data)
visual(avg_pat)


ctr_ori,r = ctrRadiusIni(avg_pat)
print('Center of the zero-order disk is {}. The radius of the disk is {}.'.format(ctr_ori,r))


fig, ax = plt.subplots(figsize = (5,5))
ax.imshow(avg_pat)
y, x = ctr_ori
c = plt.Circle((x, y),r, color='red', linewidth=2, fill=False)
ax.add_patch(c)
plt.tight_layout()
plt.show()


kernel = genrateKernel(avg_pat,ctr_ori,r,0.7,1)
visual(kernel)

kernel[kernel < kernel.mean()] = 0
kernel[kernel !=0] = 1
visual(kernel)


cros_map = crossCorr(avg_pat,kernel)


blobs = ctrDet(cros_map, r, kernel, 10, 5)

drawCircles(avg_pat,blobs,r)


ref_ctr = radGradMax(avg_pat, blobs, r,ra=4)

ref_blobs_list = ref_ctr[:,:2]

drawCircles(avg_pat,ref_blobs_list,r)


# Detect angle and renew weights
angle, refined_ctr = detAng(ref_ctr,ctr_ori,r)

print('Estimated rotation angle: ',angle,'(deg)')


# Generate the coordinate of centers in the new coordinate system 
rot_ref_ctr = rotCtr(avg_pat,ref_ctr,angle)  


refe_a,refe_b,ref_ctr2, mid_ctr,ref_ang = latFit(avg_pat,rot_ref_ctr,r)

print('Two lattice vectors: vector_a--[',refe_a[0],refe_a[1], '] and vector_b--[',refe_b[0],refe_b[1],']')


gen_lat_pt = genLat(avg_pat, refe_a, refe_b, mid_ctr,r)


result_pt = delArti(gen_lat_pt,ref_ctr2,r)
test = rotImg(avg_pat, angle+ref_ang, ctr_ori)
drawCircles(test,result_pt,r)

a_init,b_init = latBack(refe_a, refe_b, angle)


ctr_new = []
ang_rad = -(angle)*np.pi/180
ctr = mid_ctr[0]

for q in range (len(result_pt)):
    cur_cd = result_pt[q,:2]
    y_new = -(ctr[0] - (cur_cd[0]-ctr[0])*np.cos(ang_rad) + (cur_cd[1]-ctr[1])*np.sin(ang_rad) ) + 2*ctr[0]
    x_new = (ctr[1] + (cur_cd[0]-ctr[0])*np.sin(ang_rad) + (cur_cd[1]-ctr[1])*np.cos(ang_rad) )

    if y_new>0 and x_new>0 and y_new<pxh and x_new<pxw:
        ctr_new.append([y_new,x_new])

ctr_new = np.array(ctr_new)
   
pat = (((avg_pat - avg_pat.min()) / (avg_pat.max() - avg_pat.min())) * 255).astype(np.uint8)

pat_rgb = cv2.cvtColor(pat,cv2.COLOR_GRAY2RGB)

vis_ctr = np.round(ctr_new,0)
vis_ctr.astype(int)

drawCircles(pat_rgb,vis_ctr,r)



lat_par = np.zeros((imgh,imgw,2,2),dtype = float)

for row_idx in range (5,imgh):#(imgh):
    for col_idx in range (imgw):
        pattern = data[row_idx,col_idx]
        
        cros_map = crossCorr(pattern,kernel)   
        blobs = ctrDet(cros_map, r, kernel, 10, 10) 
        
        # drawCircles(pattern,blobs,r)
        
        if len(blobs) > 5:
            ctr_cur,r_cur = ctr_radius_ini(pattern)
            if np.linalg.norm(ctr_cur-ctr_ori) <= 2: # 2px
                ctr = ctr_cur
                # print(ctr)
            else:
                ctr = ctr_ori
                ctr[1] = round(ctr[1])
                ctr[0] = round(ctr[0])
                
            
            ref_ctr = radGradMax(pattern, blobs, r,rn=20, ra=4, n_p=40, threshold=3)
            # drawCircles(pattern,ref_ctr[:,:2],r)
            
            
            ctr_vec = ref_ctr[:,:2] - ctr
            ctr_diff = ctr_vec[:,0]**2 + ctr_vec[:,1]**2
            ctr_idx = np.where(ctr_diff==ctr_diff.min())[0][0]
            ref_ctr[ctr_idx,2] = 10**38
            rot_ref_ctr = rotCtr(pattern,ref_ctr,angle)
            ret_a,ret_b,ref_ctr2, mid_ctr,ref_ang = latFit(pattern,rot_ref_ctr,r)
            
            if any(ret_a!=0) and any(ret_b!=0):
                a_back,b_back = latBack(ret_a, ret_b, angle+ref_ang)             
                
                lat_par[row_idx,col_idx,0,:] = a_back
                lat_par[row_idx,col_idx,1,:] = b_back                
   
            #     gen_lat_pt = genLat(pattern, ret_a, ret_b, mid_ctr,r)
            #     result_pt = delArti(gen_lat_pt,ref_ctr2,r)
               
                
            #     ctr_new = []
            #     ang_rad = -(angle+ref_ang)*np.pi/180

            #     for q in range (len(result_pt)):
            #         cur_cd = result_pt[q,:2]
            #         y_new = -(ctr[0] - (cur_cd[0]-ctr[0])*np.cos(ang_rad) + (cur_cd[1]-ctr[1])*np.sin(ang_rad) ) + 2*ctr[0]
            #         x_new = (ctr[1] + (cur_cd[0]-ctr[0])*np.sin(ang_rad) + (cur_cd[1]-ctr[1])*np.cos(ang_rad) )

            #         if y_new>0 and x_new>0 and y_new<pxh and x_new<pxw:
            #             ctr_new.append([y_new,x_new])

            #     ctr_new = np.array(ctr_new)  
   
            # ##########################
            #     pat = (((pattern - pattern.min()) / (pattern.max() - pattern.min())) * 255).astype(np.uint8)

            #     pat_rgb = cv2.cvtColor(pat,cv2.COLOR_GRAY2RGB)

            #     vis_ctr = np.array(ctr_new,dtype=int)
                
            #     drawCircles(pat_rgb,vis_ctr,r)
              
                

    print('Processed {} out of {} rows of patterns.'.format(row_idx,imgh))
    
print('-------------------Process Finished-------------------')

###############################################################################


lat_fil = latDist(lat_par,a_init,b_init)

st_xx,st_yy,st_xy,st_yx,tha_ang = calcStrain(lat_fil, a_init,b_init)



rdbu = plt.cm.get_cmap('RdBu')
cmap_re = rdbu.reversed()
  
input_min=-0.058
input_max=0.058

l_min = input_min*100
l_max = input_max*100

titles = ["$\epsilon_{xx}(\%)$","$\epsilon_{yy}(\%)$","$\epsilon_{xy}(\%)$","$\Theta$"]
comb = [st_xx,st_yy,st_xy,tha_ang]


fig,axs = plt.subplots(2,2,figsize = (15,15))
i=0
for row in range (2):
    for col in range (2):
        if row==1 and col==1:
            ax = axs[row,col]
            pcm = ax.imshow(comb[i],cmap=cmap_re,vmin=l_min,vmax=l_max)
            ax.set_title(titles[i],fontsize=30)
            ax.set_axis_off()
            fig.colorbar(pcm,ax=ax)
            i +=1
        else:    
            ax = axs[row,col]
            pcm = ax.imshow(comb[i]*100,cmap=cmap_re,vmin=l_min,vmax=l_max)
            ax.set_title(titles[i],fontsize=30)
            ax.set_axis_off()
            fig.colorbar(pcm,ax=ax)
            i +=1
       
plt.subplots_adjust(wspace=0.25,hspace=0.25)
plt.show()

