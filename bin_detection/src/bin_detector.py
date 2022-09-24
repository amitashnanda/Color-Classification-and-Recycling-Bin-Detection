'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

# from os import O_TEMPORARY
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,			
			e.g., parameters of your classifier
		'''
		self.theta_bb, self.theta_nbb, self.theta_lc, self.theta_dc = np.loadtxt('bin_detection/theta_TD_rgb.txt')
		self.mu_bb, self.mu_nbb, self.mu_lc, self.mu_dc = np.loadtxt('bin_detection/mu_TD_rgb.txt')
		self.cov_bb, self.cov_nbb, self.cov_lc, self.cov_dc = np.loadtxt('bin_detection/cov_bb_TD_rgb.txt'),np.loadtxt('bin_detection/cov_nbb_TD_rgb.txt'), np.loadtxt('bin_detection/cov_lc_TD_rgb.txt'), np.loadtxt('bin_detection/cov_dc_TD_rgb.txt')

		# self.theta_bb, self.theta_nbb, self.theta_lc, self.theta_dc = np.loadtxt('theta_TD_rgb.txt')
		# self.mu_bb, self.mu_nbb, self.mu_lc, self.mu_dc = np.loadtxt('mu_TD_rgb.txt')
		# self.cov_bb, self.cov_nbb, self.cov_lc, self.cov_dc = np.loadtxt('cov_bb_TD_rgb.txt'),np.loadtxt('cov_nbb_TD_rgb.txt'), np.loadtxt('cov_lc_TD_rgb.txt'), np.loadtxt('cov_dc_TD_rgb.txt')
		
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		# img = cv2.imread(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		#segm_img = cv2.imread(img)
		#segm_rgb_img = cv2.cvtColor(segm_img, cv2.COLOR_BGR2RGB)
		# segm_yuv_img = cv2.cvtColor(segm_img, cv2.COLOR_BGR2YUV)
		s = np.shape(img)
		d = s[0]*s[1]
		#y = np.zeros((s[0],s[1],3))
		y = np.zeros((s[0],s[1]))

		d_bb = np.linalg.det(self.cov_bb)
		d_nbb = np.linalg.det(self.cov_nbb)
		d_lc = np.linalg.det(self.cov_lc)
		d_dc = np.linalg.det(self.cov_dc)

		i_bb = np.linalg.inv(self.cov_bb)
		i_nbb = np.linalg.inv(self.cov_nbb)
		i_lc = np.linalg.inv(self.cov_lc)
		i_dc = np.linalg.inv(self.cov_dc)

		for i in range(s[0]):
			for j in range(s[1]):
				pdf_bb = np.log(self.theta_bb) - ( np.log(2*np.pi*d_bb) )/2 - ( np.dot(np.dot( ((img[i,j]/255 - self.mu_bb).T), i_bb ) , (img[i,j]/255 - self.mu_bb)) )/2
				pdf_nbb = np.log(self.theta_nbb) - ( np.log(2*np.pi*d_nbb) )/2 - ( np.dot(np.dot( ((img[i,j]/255 - self.mu_nbb).T), i_nbb ) , (img[i,j]/255 - self.mu_nbb)) )/2
				pdf_lc = np.log(self.theta_lc) - ( np.log(2*np.pi*d_lc) )/2 - ( np.dot(np.dot( ((img[i,j]/255 - self.mu_lc).T), i_lc ) , (img[i,j]/255 - self.mu_lc)) )/2
				pdf_dc = np.log(self.theta_dc) - ( np.log(2*np.pi*d_dc) )/2 - ( np.dot(np.dot( ((img[i,j]/255 - self.mu_dc).T), i_dc ) , (img[i,j]/255 - self.mu_dc)) )/2

				if np.argmax([pdf_bb, pdf_nbb, pdf_lc, pdf_dc]) == 0:
					y[i,j] = 1
				else:
					y[i,j] = 0

		# plt.imshow(y)
		# plt.show()
		
		
		# lbl = label(y) 
		# props = regionprops(lbl)
		# for prop in props:
		# 	print('Found bbox', prop.bbox)
		# 	cv2.rectangle(segm_rgb_img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)
		
		# plt.imshow(segm_rgb_img)
		# plt.show()
		# Replace this with your own approach 
		#mask_img = img
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
	
		# img = img.astype(np.uint8)
		# contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		# box = []
		# for c in contours:
		# 	x,y,w,h = cv2.boundingRect(c)
		# 	# area = cv2.contourArea(c)
		# 	ar = float(w/h)
		# 	# ext = float(area)/(w*h)

		# 	if w/h>1.25 or w/h<0.3 or w*h<4000:
		# 		continue
		# 	if ar<1.1 or w*h>0.6*(img.shape[0]*img.shape[1]):
		# 		# cv2.rectangle(img,x,y)
		# 		box.append([x,y,x+w,y+h])

		#img = cv2.imread(os.path.join(folder,filename))
		# img = cv2.imread(img)
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		#mask = self.segment_image(img)
		
		lbl = label(img) 
		props = regionprops(lbl)
		box = []
		
		#segm_img = cv2.imread(img)
		#segm_rgb_img = cv2.cvtColor(segm_img, cv2.COLOR_BGR2RGB)
			
		for prop in props:
			h =abs(prop.bbox[0]-prop.bbox[2])
			w = abs(prop.bbox[1]-prop.bbox[3])
			# if w/h>1.25 or w/h<0.3 or w*h<3000:
				# continue
			if w/h>1.25 or w/h<0.3 or w*h<3000:
				continue
			if w/h<1.1 or 200000>w*h>10000:	


				t = list(prop.bbox)
				t[0], t[1] = t[1], t[0]
				t[2], t[3] = t[3], t[2]
				box.append(t)



			#cv2.rectangle(img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)	
			
		#print(box)
		# plt.imshow(img)
		# plt.show()
        

		
		# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		# cv.drawContours(img, contours, -1, (0,255,0), 3)
		
		# Replace this with your own approach 
		# x = np.sort(np.random.randint(img.shape[0],size=2)).tolist()
		# y = np.sort(np.random.randint(img.shape[1],size=2)).tolist()
		# boxes = [[x[0],y[0],x[1],y[1]]]
		# boxes = [[182, 101, 313, 295]]


		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		

		return box

	# def generate_rgb_values(self):

	def train_rgb_values(self):

		colour_list = ['rgb', 'yuv']

		for i in colour_list:

			# folder = 'data/training'
			X1 = np.loadtxt('blue_bin_'+i+'.txt', dtype = float)
			X2 = np.loadtxt('non_bin_blue_'+i+'.txt', dtype = float)
			X3 = np.loadtxt('light_colour_'+i+'.txt', dtype = float)
			X4 = np.loadtxt('dark_colour_'+i+'.txt', dtype = float)

			y1, y2, y3,y4 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3), np.full(X4.shape[0],4)
			# X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3,y4))



			# len_r = len(X1)
			# len_g = len(X2)
			# len_b = len(X3)
			# total_length = len(X1)+len(X2)+len(X3)

			# y1 = list()
			# y2 = list()
			# y3 = list()
			# y4 = list()
			# for i in range(len(y)):
			# 	if y[i]==1:
			# 		y1.append(y[i])
			# 	elif y[i] == 2:
			# 		y2.append(y[i])
			# 	elif y[i] == 3:
			# 		y3.append(y[i])

			# theta_r, theta_g, theta_b = len_r/total_length, len_g/total_length, len_b/total_length
			tot_len_y = len(y1)+len(y2)+len(y3)
			theta_bb, theta_nbb, theta_lc, theta_dc = len(y1)/tot_len_y, len(y2)/tot_len_y, len(y3)/tot_len_y, len(y4)/tot_len_y
			mu_bb, mu_nbb, mu_lc, mu_dc = np.average(X1,axis = 0), np.average(X2,axis = 0), np.average(X3,axis = 0),np.average(X4,axis = 0) 
			cov_bb, cov_nbb, cov_lc, cov_dc  = np.cov(X1.T), np.cov(X2.T), np.cov(X3.T),np.cov(X4.T)


			np.savetxt('theta_TD_'+i+'.txt', [theta_bb, theta_nbb, theta_lc, theta_dc], fmt = "%s")
			np.savetxt('mu_TD_'+i+'.txt', [mu_bb, mu_nbb, mu_lc, mu_dc], fmt = "%s")
			np.savetxt('cov_bb_TD_'+i+'.txt', cov_bb, fmt = "%s")
			np.savetxt('cov_nbb_TD_'+i+'.txt', cov_nbb, fmt = "%s")
			np.savetxt('cov_lc_TD_'+i+'.txt', cov_lc, fmt = "%s")
			np.savetxt('cov_dc_TD_'+i+'.txt', cov_dc, fmt = "%s")

			pass

# class_call = BinDetector()
# # # # class_call.train_rgb_values()

# class_call.get_bounding_boxes('data/validation/0061.jpg')












