import tensorflow as tf
import numpy as np
class convNet():

	def weight_variable(self, shape):
		    #The optimiser can get stuck in its initial position if you do not truncate. 
    		#tf.truncated_normal is a TensorFlow function that produces random values following the
    		#normal (Gaussian) distribution between -2*stddev and +2*stddev.
		return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

	def bias_variable(self, length):
	    return tf.Variable(tf.constant(0.01, shape = length))

	def max_pool_2x2(self, x):
	    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

	def conv2d(self, x, W, stride):
	    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

	def createNetwork(self, actions, X_AXIS, Y_AXIS):
	    #Create the convolutional neural network
    	#print("Creating the network...")
    	#The input of W_conv1 consists of 4x80x80 images
    	#W_conv1 convolves 32 filters of 8 x 8 with a stride of 4 	
	    self.W_conv1 = self.weight_variable([8, 8, 4, 32])
	    self.b_conv1 = self.bias_variable([32])

	    self.W_conv2 = self.weight_variable([4, 4, 32, 64])
	    self.b_conv2 = self.bias_variable([64])

	    self.W_conv3 = self.weight_variable([3, 3, 64, 64])
	    self.b_conv3 = self.bias_variable([64])

	    self.W_fc4 = self.weight_variable([576, 512])
	    self.b_fc4 = self.bias_variable([512])

	    self.W_fc5 = self.weight_variable([512, actions])
	    self.b_fc5 = self.bias_variable([actions])

	    # input layer
	    self.s = tf.placeholder("float", [None, X_AXIS, Y_AXIS, 4])

	    # hidden layers
	    self.h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
	    self.h_pool1 = self.max_pool_2x2(self.h_conv1)
	    
	    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)
	    self.h_pool2 = self.max_pool_2x2(self.h_conv2)
	    #The third layer convolves a 64 filter of 3x3 with a stride of 1
	    self.h_conv3 = tf.nn.relu(self.conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
	    self.h_pool3 = self.max_pool_2x2(self.h_conv3)
	    #Reshape flattens the game image into a single vector (80 pixels wide* 80 pixels height = 1600 pixels total)
	    #-1 means start at the position that captures the entire image
	    self.h_conv3_flat = tf.reshape(self.h_pool3, [-1, 576])

	    #final hidden activations
	    self.h_fc4 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4) + self.b_fc4)
	    #The output layer is fully connected with a single output for each valid action
	    self.readout = tf.matmul(self.h_fc4, self.W_fc5) + self.b_fc5
	    #Ylogits = tf.matmul(h_fc4, W_fc5) + b_fc5
	    return self.s, self.readout, self.h_fc4

