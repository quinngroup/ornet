import numpy as np
import sys
import matplotlib.pyplot as plt
import os.path
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob



def Read_files(c_path, l_path, m_path, extensions):

	control_filenames = []
	llo_filenames = []
	mdivi_filenames = []


	# We assume that we have different directories for cont , llo and mdivi numpy arrays , thus, 
	# we extracts all the .npy files and save them in c, l and m lists . 
	# Also we should exclude ".DS_Store"  files ===>  " ... (not(f.startwith('.'))) ..."
	c = []
	l = []
	m = []

	onlyfiles_c = [f for f in listdir(c_path) if (isfile(join(c_path, f)) and (not (f.startswith('.'))) and (f.endswith('.npy')))]
	onlyfiles_l = [f for f in listdir(l_path) if (isfile(join(l_path, f)) and (not (f.startswith('.'))) and (f.endswith('.npy')))]
	onlyfiles_m = [f for f in listdir(m_path) if (isfile(join(m_path, f)) and (not (f.startswith('.'))) and (f.endswith('.npy')))]


	control_filenames.extend( sorted(glob.glob(os.path.join( c_path, "*.npy"))))
	llo_filenames.extend( sorted(glob.glob(os.path.join( l_path, "*.npy"))))
	mdivi_filenames.extend( sorted(glob.glob(os.path.join( m_path, "*.npy"))))

	print(len(control_filenames),len(llo_filenames),len(mdivi_filenames))

	for i in range (len(llo_filenames)):
	#     #print (control_filenames[i])
	   print (llo_filenames[i])

	for files in range (len(onlyfiles_c)):
	    c.append(np.load(control_filenames[files]))
	    
	for files in range (len(onlyfiles_l)):
	    l.append(np.asarray(np.load(llo_filenames[files])))
	    
	for files in range (len(onlyfiles_m)):
	    m.append(np.load(mdivi_filenames[files]))
	print (len(c), len(l), len(m))

	cont = np.asarray(c)
	llo = np.asarray(l)
	mdivi = np.asarray(m)
	return cont, llo, mdivi

def shadow_thresholding( files, thresh):
	new_files_list = []
	for videos in range (files.shape[0]):
	    new_file = []
	    for frames in range (files[videos].shape[0]):
	        img = files[videos][frames]
	        img[ img <= thresh ] = 0
	        new_file.append(img)
	    new_files_list.append(new_file)
	new_files_arr = np.asarray(new_files_list)
	print(new_files_arr.shape)
	return(new_files_arr)


def normalization_proc (files):
	new_norm_list = []
	for videos in range (files.shape[0]):
		new_norm = []
		for frames in range (files[videos].shape[0]):
			img = files[videos][frames]
			base = np.min(img)
			range_intens = np.max(img) - base
			normalized = [(x-base)/range_intens for x in img]
			new_norm.append(normalized)
		new_norm_list.append(np.asarray(new_norm))
	new_norm_arr = np.asarray(np.asarray(new_norm_list))
	print(new_norm_arr.shape, new_norm_arr[0].shape)
	return new_norm_arr

def max_medians_list(files):
	medians_list = []
	max_list = []
	for videos in range (files.shape[0]):
	    medians_l = []
	    for frames in range (files[videos].shape[0]):
	        m = files[videos][frames].flatten()
	        k = m[m>0]
	        medians_l.append(np.median(k))
	    max_list.append(np.max(medians_l))
	    medians_list.append(np.asarray(medians_l))
	all_medians = np.asarray(medians_list)
	#maxx = np.max(max_list)
	maxx = np.asarray(max_list)
	return maxx, all_medians

def compute_difference_mat(all_medians, maxx):
	mm_diff_l = []
	for video in range (all_medians.shape[0]):
	    mm_diff = []
	    for frame in range(all_medians[video].shape[0]):
	        mm_diff.append(maxx[video] - all_medians[video][frame])
	    mm_diff_l.append(np.asarray(mm_diff))
	mm_diff_l = np.asarray(mm_diff_l)
	return mm_diff_l

def median_equalization(files, medians_list, differences, prefix):
	for video in range(files.shape[0]):
		new_files_tmp = []
		for frame in range(medians_list[video].shape[0]):
			img = files[video][frame]
			diff = int(differences[video][frame])
			array = img.flatten()
			new_l = [item + diff if item != 0 else 0 for item in array]
			new_l = np.array(new_l).reshape(512,512)
			new_files_tmp.append(new_l)
		print(video)
		new_files_arr = np.asarray(new_files_tmp)
		np.save( prefix + str(video) +'.npy', new_files_arr)
		print('new numpy file is created...')


exts =["*.npy"]

c_path ='/Users/mojtaba/Desktop/OrNet Project/single_vids/control'
l_path ='/Users/mojtaba/Desktop/OrNet Project/single_vids/llo' 
m_path ='/Users/mojtaba/Desktop/OrNet Project/single_vids/mdivi'

numpy_file_prefix_ll = 'llo__'
numpy_file_prefix_mm = 'mdv__'
numpy_file_prefix_cc = 'cont__'

cont, llo, mdivi = Read_files(c_path, l_path, m_path, exts)


# new_cont = shadow_thresholding(cont, 3)
# new_llo = shadow_thresholding(llo, 3)
# new_mdvi = shadow_thresholding(mdivi, 3)


# new_c_arr = normalization_proc(new_cont)
# new_l_arr = normalization_proc(new_llo)
# new_m_arr = normalization_proc(new_mdvi)

maxx_c, cc = max_medians_list(cont)
maxx_l, ll = max_medians_list(llo)
maxx_m, mm = max_medians_list(mdivi)

diff_c = compute_difference_mat(cc, maxx_c)
diff_l = compute_difference_mat(ll, maxx_l)
diff_m = compute_difference_mat(mm, maxx_m)
print (maxx_c, diff_m.shape)
print ('now computing and generating the numpy files...')

median_equalization(cont, cc, diff_c, numpy_file_prefix_cc)
median_equalization(llo, ll, diff_l, numpy_file_prefix_ll)
median_equalization(mdivi, mm, diff_m, numpy_file_prefix_mm)

