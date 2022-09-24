'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


from operator import ge
from statistics import mode
import numpy as np
import os

# file_name = 'generate_rgb_data.py'

# folder_path = os.path.dirname(os.path.abspath(file_name)) 
# # print(folder_path)
# model_params_file = os.path.join(folder_path,'pixel_classifier.py')
# # print(model_params_file)


from pixel_classification.generate_rgb_data import read_pixels


class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
      
    '''
    # self.theta_r, self.theta_g, self.theta_b  = np.loadtxt('theta_TD.txt')
    # self.mu_r, self.mu_g, self.mu_b = np.loadtxt('mu_TD.txt')
    # self.cov_r, self.cov_g, self.cov_b = np.loadtxt('cov_r_TD.txt'), np.loadtxt('cov_g_TD.txt'), np.loadtxt('cov_b_TD.txt')
    
    self.theta_r = 0.36599891716296695
    self.theta_g = 0.3245804006497022
    self.theta_b = 0.3094206821873308

    self.mu_r = np.array([0.752506091193876, 0.3480856247824596, 0.34891228680821507])
    self.mu_g = np.array([0.35060916777052853, 0.7355148898592007, 0.32949353219186056])
    self.mu_b = np.array([0.3473590311015048, 0.33111351277169027, 0.7352649546257726])

    self.cov_r = np.array([[0.037086702210567715, 0.01844078389423107, 0.018632848266471724],
                          [0.01844078389423107, 0.062014562077284396, 0.00858163574699616],
                          [0.018632848266471724, 0.00858163574699616, 0.06206845784048821]])
    self.cov_g = np.array([[0.0557811526944266, 0.017653267296060524, 0.008739554602480014],
                          [0.017653267296060524, 0.03481496393718223, 0.017023401075060354],
                          [0.008739554602480014, 0.017023401075060354, 0.056068642903266]])
    self.cov_b = np.array([[0.05458537840583401, 0.008552820244218098, 0.017173502589262955],
                          [0.008552820244218098, 0.056883076264007806, 0.0183084868818329],
                          [0.017173502589262955, 0.0183084868818329, 0.035771903528807804]])

  

    

    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
   

    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Just a random classifier for now
    # Replace this with your own approach 
    y = np.empty(len(X))

    d_r = np.linalg.det(self.cov_r)
    d_g = np.linalg.det(self.cov_g)
    d_b = np.linalg.det(self.cov_b)

    i_r = np.linalg.inv(self.cov_r)
    i_g = np.linalg.inv(self.cov_g)
    i_b = np.linalg.inv(self.cov_b)

    
    for i in range (len(X)):
      pdf_r = np.log(self.theta_r) - ( np.log(2*np.pi*d_r) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_r).T), i_r ) , (X[i,:] - self.mu_r)) )/2
      pdf_g = np.log(self.theta_g) - ( np.log(2*np.pi*d_g) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_g).T), i_g ) , (X[i,:] - self.mu_g)) )/2
      pdf_b = np.log(self.theta_b) - ( np.log(2*np.pi*d_b) )/2 - ( np.dot(np.dot( ((X[i,:] - self.mu_b).T), i_b ) , (X[i,:] - self.mu_b)) )/2

      max_val = max(pdf_r, pdf_g, pdf_b)

      if(max_val == pdf_r):
        y[i] = 1
      elif(max_val == pdf_g):
        y[i] = 2
      elif(max_val == pdf_b):
        y[i] = 3

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

  def train(self):

    folder = 'data/training'
    X1 = read_pixels(folder+'/red')
    X2 = read_pixels(folder+'/green')
    X3 = read_pixels(folder+'/blue')
    y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
    X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))



    # len_r = len(X1)
    # len_g = len(X2)
    # len_b = len(X3)
    # total_length = len(X1)+len(X2)+len(X3)

    y1 = list()
    y2 = list()
    y3 = list()
    for i in range(len(y)):
        if y[i]==1:
            y1.append(y[i])
        elif y[i] == 2:
            y2.append(y[i])
        elif y[i] == 3:
            y3.append(y[i])

    # theta_r, theta_g, theta_b = len_r/total_length, len_g/total_length, len_b/total_length
    theta_r, theta_g, theta_b = len(y1)/len(y), len(y2)/len(y), len(y3)/len(y)
    mu_r, mu_g, mu_b = np.average(X1,axis = 0), np.average(X2,axis = 0), np.average(X3,axis = 0) 
    cov_r, cov_g, cov_b = np.cov(X1.T), np.cov(X2.T), np.cov(X3.T)


    np.savetxt('theta_TD.txt', [theta_r, theta_g, theta_b], fmt = "%s")
    np.savetxt('mu_TD.txt', [mu_r, mu_g, mu_b], fmt = "%s")
    np.savetxt('cov_r_TD.txt', cov_r, fmt = "%s")
    np.savetxt('cov_g_TD.txt', cov_g, fmt = "%s")
    np.savetxt('cov_b_TD.txt', cov_b, fmt = "%s")

    pass









    
    
   


     

    



