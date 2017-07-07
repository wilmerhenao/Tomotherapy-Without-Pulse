
# coding: utf-8

# In[1]:


import matplotlib.colors as col
import matplotlib.pyplot as plt


import numpy as np
# get_ipython().magic(u'matplotlib inline')

import scipy.sparse as sps



# In[2]:


base_dir = '123456 -Helical Gyn/'
# base_dir = 'prostate/'

# img_filename = 'roimask.img'
# header_filename = 'roimask.header'
img_filename = 'samplemask.img'
header_filename = 'samplemask.header'
struct_img_filename = 'roimask.img'
struct_img_header = 'roimask.header'



# In[3]:
def getvector(necfile,dtype):
    with open(necfile, 'rb') as f:
        try:
            data = np.fromfile(f, dtype=dtype)
        finally:
            f.close()
    return(data)


def get_dim(base,fname):
    with open(base+fname,'r') as header:
        dim_xyz = [0]*3
        for i,line in enumerate(header):
            if 'x_dim' in line:
                dim_xyz[0] = int(line.split(' ')[2])
            if 'y_dim' in line:
                dim_xyz[1] = int(line.split(' ')[2])
            if 'z_dim' in line:
                dim_xyz[2] = int(line.split(' ')[2])
    return dim_xyz


# In[4]:

def get_img_array(base,fname, dim_xyz,dtype=np.uint16):
    with open(base_dir+img_filename,'rb') as f:
        img_arr = np.fromfile(f,dtype=dtype)
        img_arr = img_arr.reshape(dim_xyz[0],dim_xyz[1],dim_xyz[2],order='F')
    return np.copy(img_arr)


# In[5]:

def get_structure_mask(struct_id_list, struct_img_arr):
        
    img_struct = np.zeros_like(struct_img_arr)
    for s in struct_id_list:    
            img_struct[np.where(struct_img_arr & 2**(s-1))] = s
    return np.copy(img_struct)

def get_subsampled_mask(struct_img_mask_full_res, subsampling_img):
    sub_sampled_img_struct = np.zeros_like(struct_img_mask_full_res)
    sub_sampled_img_struct[np.where(subsampling_img)] =  struct_img_mask_full_res[np.where(subsampling_img)]
    return np.copy(sub_sampled_img_struct)

def plot_structure(img,struct_id_list,zslices,printSliceCount=False, plotstuff=True, struct_num_max=16):
    img_struct = np.zeros_like(img)
   
    for s in struct_id_list:
    
        img_struct[np.where(img_arr & 2**(s-1))] = s

                  
    count = 0
    if isinstance(zslices,int):
        num_slices = 1
        if plotstuff:
            plt.imshow(img_struct[:,:,zslices],cmap='gray',interpolation='nearest')
            plt.show()
    else:
        num_slices = zslices.size
        for z in zslices:
            if np.any(img_struct[:,:,z]>0):
                count+=1
            if plotstuff:
                plt.imshow(img_struct[:,:,z],cmap='gray',interpolation='nearest')
                plt.show()
    if printSliceCount:
        print 'Structure(s) {} is in {} of {} plotted slices'.format(str(struct_id_list),count,num_slices)
        
        

# plot_structure(img_arr,np.array([1,3,7,10]),np.arange(60,65),printSliceCount=True,plotstuff=True)

if base_dir=='prostate/':
    dtype=np.uint32
else:
    dtype=np.uint16

dim_xyz = get_dim(base_dir, header_filename)
# get subsample mask
img_arr = get_img_array(base_dir, img_filename, dim_xyz, dtype=dtype)
# get structure file
struct_img_arr = get_img_array(base_dir, struct_img_filename, dim_xyz, dtype=dtype)
print struct_img_arr.shape, np.count_nonzero(struct_img_arr), 'full struct img shape, number nonzero vox'

struct_id_list = [i for i in range(1,22)]
# struct_id_list = [i for i in range(1,2)]
img_struct = get_structure_mask(struct_id_list, struct_img_arr)

sub_sampled_img_struct = get_subsampled_mask(img_struct, img_arr)

print sub_sampled_img_struct.shape, np.count_nonzero(sub_sampled_img_struct), 'subsampled img shape, number nonzero vox'
print struct_img_arr.shape, np.count_nonzero(struct_img_arr), 'full struct img shape, number nonzero vox'
struct_img_arr
bixels = getvector(base_dir + 'dij/Bixels_out.bin', np.int32)
voxels = getvector(base_dir + 'dij/Voxels_out.bin', np.int32)
Dijs = getvector(base_dir + 'dij/Dijs_out.bin', np.float32)
print bixels.shape, bixels.min(), bixels.max(), 'bix shape, bix min, bix max'
print voxels.shape, voxels.min(), voxels.max(), 'vox'
print Dijs.shape, Dijs.min(), Dijs.max(), 'dij'
print (np.unique(voxels)).size, 'num unique voxels'

flatNP = sub_sampled_img_struct.flatten(order='F')
print np.unique((flatNP[np.unique(voxels)])), (flatNP[np.unique(voxels)]).size, 'vox getting radiation'
print np.sum(sub_sampled_img_struct[sub_sampled_img_struct>0]), sub_sampled_img_struct.shape, sub_sampled_img_struct.size, 'subsampled body mask vox count, img shape, img size'



print img_arr[img_arr>0].size

# plt.imshow(img_arr[:,:,80],cmap='gray',interpolation='nearest')
# plt.show()

plt.imshow(sub_sampled_img_struct[:,:,80],cmap='gray',interpolation='nearest')
plt.show()
plotter = np.zeros_like(struct_img_arr) 
plotter[np.where(img_arr)] = struct_img_arr[np.where(img_arr)]

print np.count_nonzero(img_struct[np.where(img_arr)]), 'number nonzero where sampled'
print plotter.shape
plt.imshow(plotter[:,:,80],cmap='gray',interpolation='nearest')
plt.show()

plt.imshow(img_struct[:,:,80],cmap='gray',interpolation='nearest')
plt.show()

plt.imshow(struct_img_arr[:,:,80],cmap='gray',interpolation='nearest')
plt.show()

# plot_structure(struct_img_arr, struct_id_list, np.array([80,82]))