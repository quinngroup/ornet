import numpy as np
import cv2
from matplotlib import pyplot as plt
import imageio
import numpy as np
import random

def main():
	# First we read the manual segmentation and copy it to 'im'
	filename = 'DsRed2-HeLa_2_21_LLO'
	foldername = 'LLO'
	im = imageio.imread('First Frames Segments/'+foldername+'/'+filename+'.vtk')#Directory will be machine dependant
	plt.imshow(im)
	plt.show()

	#=======================================================================#
	#																		#
	#=======================================================================#

	video_file ='Videos/'+foldername+'/'+filename+' (Converted).mov' #Directory is machine dependant
	vf = cv2.VideoCapture(video_file)	#OpenCV videocapture object

	# obtaining the frame dimensions
	X = int(vf.get(3)) #vf.get returns specific properties, 3 is width 4 is height
	Y = int(vf.get(4))

	print (X, Y)

	# Setting the variables for saving the out put video like File name , encoding method , fps and ...
	fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
	out = cv2.VideoWriter('test1.mov',fourcc, 30.0, (X, Y)) #OpenCV VideoWriter Object
	


	frameNum = 0
	number_of_segments = len(np.unique(im))-1 #defines number of segs from vtk
	print(number_of_segments)
	outs = list()
	for i in range(number_of_segments):
		outs.append(cv2.VideoWriter(filename+'_Cell'+str(i)+'.mov',fourcc,30.0, (X, Y)))

	ims = list () #holds masked images
	masks = list() #holds masks
	opening = list() #holds images after opened
	colors = list() #the colors to make the contours in the output video?
	contours = list() #holds contours
	dilates = list() #holds dilated shapes
	outframes = list()
	first = True #determines first frame
	kernel = np.ones((17,17), np.uint8) #kernel for opening
	kernel2 = np.ones((3,3),np.uint8) #kernel for dilation
	font = cv2.FONT_HERSHEY_SIMPLEX #font for frame count

	for cols in range (number_of_segments): #creates random colors to use for the outlines
		colors.append((random.randint(1,255), random.randint(0,255), random.randint(1,255)))


	while( vf.isOpened() ):
		#for i in range(3):#skips to every third frame
		frameNum += 1 #advance through frames
		ret, frame = vf.read()
	
		segsGrow = np.ones(number_of_segments,dtype = bool)
		for i in range (number_of_segments): #adds a copy of the current frame for each segment
			ims.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
			outframes.append(np.copy(frame))
		for i in range ( number_of_segments ): #separates each mask from the vtk and lists them
			masks.append( im != i + 1 )
		for i in range (number_of_segments): #blacks out all that isn't in the initial mask for each segment
			if(first):
				ims[i][masks[i]] = 0
			else:
				ims[i][dilates[i]!=255] = 0 #uses previous dilation for next frame
			
		del dilates[:]
		for i in range(number_of_segments): #thresholds the image with OTSU's optimal threshold value
			#ims[i] = cv2.GaussianBlur(ims[i],(5,5),0) 
			ret , ims[i] = cv2.threshold( ims[i], 6, 255, cv2.THRESH_BINARY_INV)# + cv2.THRESH_OTSU)
			#print(ret)
		for i in range (number_of_segments): #opens image to get rid of small holes
			opening.append(cv2.morphologyEx( ims[i], cv2.MORPH_OPEN, kernel, iterations = 3)) #opens image
			temp = cv2.bitwise_not(opening[i])
			dilates.append(temp) #build list dilates to iteratively dilate until overlap

		#each segment dilates
		#if two segments overlap cut the overlapping portions from both (first one to cover a place gets it...)
		for i in range (5):
			for d in range (number_of_segments):
				dilates[d] = cv2.dilate(dilates[d],kernel2,iterations=2)
			for s in range (number_of_segments): #tests each segment with every other segment, slow
				for t in range(s+1, number_of_segments):
					b_and = cv2.bitwise_and(dilates[s],dilates[t])
					if(len(np.unique(b_and)) > 1):
						dilates[s] = cv2.bitwise_xor(dilates[s],b_and)
						dilates[t] = cv2.bitwise_xor(dilates[t],b_and)
						


		for conts in range (number_of_segments): #bulds the contours and draws them onto the frame
			
			contours.append(cv2.findContours( dilates[conts], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1])
			#cont = cv2.findContours( opening[conts], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
			#if(conts == 4):
			#	cv2.fillPoly(frame,contours[conts],colors[0])
			#	cv2.imshow("test1",frame) 
			#contours.append(cont)
			cv2.drawContours( frame, contours[conts], -1, colors[conts], 2)
			if ( conts == number_of_segments - 1 ) : #prints contours frame by frame
				cv2.putText(frame, 'Frame # '+ str(frameNum), (10, 40), font, 0.5, (0, 255, 50), 1)
				cv2.imshow( "Keypoints2", frame)
				if(frameNum == 1):
					cv2.imshow("frame1", frame)
		k = cv2.waitKey(10)

		if (k == 27):
			break
		
		out.write(frame) #writes frame to video
		for i in range(number_of_segments):
			outframes[i][dilates[i]!=255] = 0
			outs[i].write(outframes[i])
		#print( frameNum)
		first = False
		del ims[:]
		del opening[:]
		del contours[:]
		del outframes[:]
		#del dilates[:]

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	out.release()
	vf.release()
if __name__ == "__main__":
	main()
