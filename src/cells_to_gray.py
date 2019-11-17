'''
This script converts the single cell videos created by
extract_helper into grayscale frames saved in a .npy
file.
'''
#Author: Marcus Hill

import os
import argparse

import cv2
import joblib
import imageio
import numpy as np

def vid_to_gray(vid_path, out_path):
	'''
	Converts the RGB video specified by the vid_path parameter into a grayscale numpy array.
	The resulting array is saved at the specified output path.

	Parameters
	----------
	vid_path: String
		path to input video
	out_path: String
		path to output directory

	Returns
	----------
	None
	'''
	vid_name = os.path.split(vid_path)[1].split('.')[0]
	frames = []
	reader = imageio.get_reader(vid_path)
	for frame in reader:
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

	reader.close()
	np.save(os.path.join(out_path, vid_name + '.npy'), frames)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description="Reads in single cell video(s) and convert them into grayscale numpy arrays.")

	#Required args
	parser.add_argument('-i', '--input', required=True, help="Path to single cell video (.avi), or directory of videos.")

	#Optional args
	parser.add_argument('-o', '--output', default=os.getcwd(), help="Path to output directory. Default cwd")
	parser.add_argument('-n','--n_jobs', default=-1, type=int, help='Number of threads to use. Default is -1 for all.')

	args = vars(parser.parse_args())
	os.makedirs(args['output'], exist_ok=True)
	vids = []
	if os.path.isdir(args['input']):
		for file_name in os.listdir(args['input']):
			if file_name.split('.')[1] in ['avi']:
				vids.append(os.path.join(args['input'],file_name))
	else:
		vids.append(args['input'])
		
	joblib.Parallel(n_jobs=args['n_jobs'], verbose=10)(
		joblib.delayed(vid_to_gray)(vid_path, args['output']) 
		for vid_path in vids)

