import numpy as np
import cv2
from matplotlib import pyplot as plt
import imageio
import numpy as np
import random

def feature_dist(input):
    """
    Takes a labeled array as returned by scipy.ndimage.label and 
    returns an intra-feature distance matrix.
    """
    I, J = np.nonzero(input)
    labels = input[I,J]
    coords = np.column_stack((I,J))

    sorter = np.argsort(labels)
    labels = labels[sorter]
    coords = coords[sorter]

    sq_dists = cdist(coords, coords, 'sqeuclidean')

    start_idx = np.flatnonzero(np.r_[1, np.diff(labels)])
    nonzero_vs_feat = np.minimum.reduceat(sq_dists, start_idx, axis=1)
    feat_vs_feat = np.minimum.reduceat(nonzero_vs_feat, start_idx, axis=0)

def main():
	
	#=======================================================================#
	#																		#
	#=======================================================================#

	video_file ='/Users/mojtaba/Desktop/OrNet Project/Videos/segmented alone the LLOs/DsRed2-HeLa_2_21_LLO_Cell0.mov' #Directory is machine dependant
	vf = cv2.VideoCapture(video_file)	#OpenCV videocapture object

	# obtaining the frame dimensions
	X = int(vf.get(3)) #vf.get returns specific properties, 3 is width 4 is height
	Y = int(vf.get(4))

	print (X, Y)

	# Setting the variables for saving the out put video like File name , encoding method , fps and ...
	fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
	out = cv2.VideoWriter('/Users/mojtaba/Desktop/OrNet Project/Videos/segmented alone the LLOs/CCL Results/connected_components.mov',fourcc, 30.0, (X, Y)) #OpenCV VideoWriter Object

	frameNum = 0
	font = cv2.FONT_HERSHEY_SIMPLEX #font for frame count


	while( vf.isOpened() ):
		#for i in range(3):#skips to every third frame
		frameNum += 1 #advance through frames
		ret, frame = vf.read()
	

		gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('gray_image',gray_image) 
		(thresh, im_bw) = cv2.threshold(gray_image, 5, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		#cv2.imshow('original', frame)
		#cv2.imshow('Thresholded',im_bw)

		ret, labels = cv2.connectedComponents(im_bw)
		#print (labels)
		flatted_labels = labels.flatten()
		Num_of_labels_pf = flatted_labels.max()
		unique, counts = np.unique(flatted_labels, return_counts=True)

		#print (np.asarray((unique, counts)).T)
		max_comp_area = counts[1:].max()
		min_comp_area = counts[1:].min()
		mean_comp_area = counts[1:].mean()
		sum_comp_area = counts[1:].sum()

		# Map component labels to hue val
		label_hue = np.uint8(179*labels/np.max(labels))
		blank_ch = 255*np.ones_like(label_hue)
		labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

		# cvt to BGR for display
		labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

		# set bg label to black
		labeled_img[label_hue==0] = 0
		cv2.putText(labeled_img, 'Frame # '+ str(frameNum), (10, 40), font, 0.5, (0, 255, 50), 1)
		cv2.putText(labeled_img, 'Number of components : '+ str(Num_of_labels_pf), (10, 58), font, 0.5, (0, 255, 50), 1)
		cv2.putText(labeled_img, 'Area of largest component : '+ str(max_comp_area), (10, 73), font, 0.5, (0, 255, 50), 1)
		cv2.putText(labeled_img, 'Area of smallest component : '+ str(min_comp_area), (10, 88), font, 0.5, (0, 255, 50), 1)
		cv2.putText(labeled_img, 'Mean area of all components : '+ str(round(mean_comp_area,2)), (10, 103), font, 0.5, (0, 255, 50), 1)
		cv2.putText(labeled_img, 'Total Area of the Cell : '+ str(sum_comp_area), (10, 125), font, 0.7, (200, 5, 150), 2)

		cv2.imshow('labeled_img',labeled_img)

		k = cv2.waitKey(10)
		if (k == 27) or frameNum > 13000:
			break

		out.write(labeled_img) #writes frame to video


	cv2.waitKey(0)
	cv2.destroyAllWindows()
	out.release()
	vf.release()
if __name__ == "__main__":
	main()