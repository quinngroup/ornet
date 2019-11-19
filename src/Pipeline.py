'''
A script that passes input video(s) through the entire OrNet pipeline. 
The pipeline consists of cell segmentation, graph vertex construction 
via Gaussian mixture model means, edge construction via divergence 
functions, and eigen-decomposition of the matrix representation.
'''
#Author: Marcus Hill

import os
import re
import argparse

import cv2
import joblib
import imageio
import numpy as np

from ornet.extract_cells import extract_cells
from ornet.extract_helper import generate_singles
from ornet.cells_to_gray import vid_to_gray
from ornet.affinityfunc import get_all_aff_tables
from ornet.gmm.run_gmm import skl_gmm

def constrain_vid(vid_path, full_path, frame_num):
	'''
	Constrains the input video to specified number of frames, and write the 
	result to an output video. If the video contains less frames than 
	frame_num, then then all frames of the video are returned.

	Parameters
	----------
	vid_path: String
		Path to the input video.
	full_path: String
		Path to the output video.
	frame_num: int
		Maximum number of frames to extract from the video.

	Returns
	----------
	NoneType object
	'''
	reader = imageio.get_reader(vid_path)
	fps = reader.get_meta_data()['fps']
	size = reader.get_meta_data()['size']
	writer = cv2.VideoWriter(full_path, 
			cv2.VideoWriter_fourcc('M','J','P','G'), 
			fps, size)

	i = 0
	for frame in reader:
		if i == frame_num:
			break
		else:
			writer.write(frame)
			i += 1

	reader.close()
	writer.release()

def cell_segmentation(vid_name, vid_path, masks_path, out_path):
	'''
	Generates segmentation masks for every frame in the video, and saves 
	the output at the specified output path.

	Parameters
	----------
	vid_path: String
		Path to input video.
	masks_path: String
		Path to initial segmentation mask.
	out_path: String
		Path to output directory.

	Returns
	----------
	NoneType object
	'''
	masks = extract_cells(vid_path, masks_path, show_video=False)
	np.save(os.path.join(out_path, vid_name + 'MASKS.npy'), masks)

def downsample_vid(vid_name, vid_path, masks_path, downsampled_path, frame_skip):
	'''
	Takes an input video and return a downsampled version of it, by 
	skipping a specified number of frames.

	Parameters
	----------
	vid_name: String
		Name of the input video.
	vid_path: String
		Path to the input video.
	masks_path: String
		Path to the input masks.
	downsampled_path:
		Path to directory where the downsampled video will be saved.
	frame_skip:
		The number of frames to skip for downsampling.

	Returns
	----------
	NoneType object
	'''
	masks = np.load(masks_path)
	masks_downsampled = [frame for i,frame in enumerate(masks) if i % 100 == 0]
	np.save(os.path.join(downsampled_path, vid_name + '.npy'), masks_downsampled)
	
	reader = imageio.get_reader(vid_path)
	fps = reader.get_meta_data()['fps']
	size = reader.get_meta_data()['size']
	writer = cv2.VideoWriter(os.path.join(downsampled_path, vid_name + '.avi'),
			cv2.VideoWriter_fourcc('M','J','P','G'),
			fps, size)
	for i,frame in enumerate(reader):
		if i % frame_skip == 0:
			writer.write(frame)
	
	reader.close()
	writer.release()

def generate_single_vids(vid_path, masks_path, output_path):
	'''
	Extracts individual cells using the segmentation masks.

	Parameters
	----------
	vid_path: String
		Path to input video.
	masks_path: String
		Path to the segmentation mask for the input video.
	output_path: String
		Directory to save the individual videos.

	Returns
	----------
	NoneType object
	'''	
	generate_singles(vid_path, masks_path, output_path)

def convert_to_grayscale(vid_path, output_path):
	'''
	Converts an input video into an array of grayscale frames
	
	Parameters
	----------
	vid_path: String
		Path to a single video.
	output_path: String
		Directory to save the grayscale frames.

	Returns
	----------
	NoneType object
	'''

	vid_to_gray(vid_path, output_path)

def compute_gmm_intermediates(vid_dir, intermediates_path):
	'''
	Generate intermediate files from passing a grayscale video 
	through the GMM portion of the pipeline.

	Parameters
	----------
	vid_dir: String
		Path to the directory that contains the single videos.
	intermediates_path:
		Path to save the intermediate files.

	Returns
	----------
	NoneType object
	'''

	file_names = os.listdir(vid_dir)
	gray_vids = [x for x in file_names if x.split('.')[-1] in ['npy']]
	
	for vid_name in gray_vids:
		try:
			vid_path = os.path.join(vid_dir,vid_name)
			vid = np.load(vid_path)
			means, covars, weights, precisions = skl_gmm(vid)
			np.savez(os.path.join(intermediates_path, vid_name.split('.')[0] + '.npz'),
				means=means, covars=covars, weights=weights, precs=precisions)
		except:
			print('Disappering cell: ' + vid_name)

def compute_distances(intermediates_path, output_path):
	'''
	Generate distances between means using Jensen-Shannon Divergence.

	Parameters
	----------
	intermediates_path: String
		Path to the GMM intermediates.
	output_path: String
		Directory to save the distance ouptuts.

	Returns
	----------
	NoneType object
	'''

	intermediates = os.listdir(intermediates_path)
	for intermediate in intermediates:
		vid_inter = np.load(os.path.join(intermediates_path, intermediate))
		table = get_all_aff_tables(vid_inter['means'], vid_inter['covars'], 'JS div')
		np.save(os.path.join(output_path, intermediate.split('.')[0] + '.npy'), 
				table)

def run(input_path, initial_masks_dir, output_path):
	'''
	Runs the entire ornet pipeline from start to finish for any video(s) 
	found at the input path location.

	Paramaters
	----------
	input_path: String
		Path to input video(s).
	initial_masks_dir: String
		Path to the directory contatining the initial segmentation mask that 
		corresponds with the input video.
	output_path: String
		Path to the output directory.

	Returns
	----------
	NoneType object
	'''

	if os.path.isdir(input_path):
		vids = [x for x in os.listdir(input_path) if x.split('.')[-1] in ['avi', 'mov']]
		input_dir = input_path
	else:
		input_dir, vids, = os.path.split(input_path)
		if vids.split('.')[-1] in ['avi', 'mov']:
			vids = [vids]
		else:
			vids = []

	if len(vids) == 0:
		print('No videos were found.')
		quit(1);

	for vid in vids:
		out_path = os.path.join(output_path, 'outputs')
		vid_name = vid.split('.')[0]
		vid_name = re.sub(' \(2\)| \(Converted\)', '', vid_name)
		full_video = os.path.join(out_path, vid_name + '.avi')
		masks_path = os.path.join(out_path, vid_name + 'MASKS.npy')
		downsampled_path = os.path.join(out_path, 'downsampled')
		singles_path = os.path.join(out_path, 'singles')
		intermediates_path = os.path.join(out_path, 'intermediates')
		distances_path = os.path.join(out_path, 'distances')
		tmp_path = os.path.join(out_path, 'tmp')

		os.makedirs(out_path, exist_ok=True)
		os.makedirs(downsampled_path, exist_ok=True)
		os.makedirs(singles_path, exist_ok =True)
		os.makedirs(intermediates_path, exist_ok =True)
		os.makedirs(distances_path, exist_ok=True)
		os.makedirs(tmp_path, exist_ok=True)

		
		constrain_vid(os.path.join(input_dir, vid), full_video, 20000)
		cell_segmentation(vid_name, full_video, 
					os.path.join(initial_masks_dir, vid_name + '.vtk'), out_path)
		downsample_vid(vid_name, full_video, masks_path, downsampled_path, 100)
		generate_single_vids(os.path.join(downsampled_path, vid_name + '.avi'),
					masks_path, tmp_path)
		single_vids = os.listdir(tmp_path)
		for single in single_vids:
			convert_to_grayscale(os.path.join(tmp_path, single), tmp_path)

		compute_gmm_intermediates(tmp_path, intermediates_path)
		compute_distances(intermediates_path, distances_path)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='An end-to-end ' 
					+ 'pipeline of OrNet.')
	parser.add_argument('-i', '--input', 
		help='Input directory containing video(s).', required=True)
	parser.add_argument('-m', '--masks', 
		help='Input directory containing vtk mask(s).', required=True)
	parser.add_argument('-o', '--output', 
		help='Output directory to save files.', default=os.getcwd())
	args = vars(parser.parse_args())
	run(args['input'], args['masks'], args['output'])
