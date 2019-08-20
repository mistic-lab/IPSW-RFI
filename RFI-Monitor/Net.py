import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from Config import Config 


class Network():
    """
    Neural Network to predict bounding boxes from 2d images
    """

    def __init__(self):
        """
        Sets up the process for the neural network.
        Defines placeholders, loss function, and the optimizer
        """
        
        self.config = Config()
        
        """EDNN implementation author: Kyle Mills""" 
        helper = EDNN_helper(L=self.config.L,f=self.config.f,c=self.config.c)

        self.x = tf.placeholder(tf.float32, [None, self.config.L, self.config.L])
        self.y = tf.placeholder(tf.float32, [None, self.config.gridN**2, 5*self.config.boxN])
        self.keep_prob = tf.placeholder(tf.float32)
        
        tiles = tf.map_fn(helper.ednn_split, self.x, back_prop=False)
        tilesT = tf.transpose(tiles, perm=[1,0,2,3,4])
        tilesF = tf.reverse(tilesT, axis = [0])
        output = tf.map_fn(self.NN, tilesF, back_prop =True)
        self.predicted = tf.transpose(output, perm=[1,0,2])
        
        
        #loss defined by position
        tmp1 = tf.square(self.predicted[:,:,1] - self.y[:,:,1])
        tmp2 = tf.square(self.predicted[:,:,2] - self.y[:,:,2])
        self.loss = tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))

        #loss defined by dimensions
        tmp1 = tf.square(self.predicted[:,:,3] - self.y[:,:,3])
        tmp2 = tf.square(self.predicted[:,:,4] - self.y[:,:,4])
        self.loss += tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))
        
        #loss defined by confidence/class selection 
        #tmp1 = tf.square(tf.where(tf.is_nan(tf.sqrt(self.predicted[:,:,3])), tf.zeros_like(self.predicted[:,:,3]), self.predicted[:,:,3]) - tf.sqrt(self.y[:,:,3]))
        #tmp2 = tf.square(tf.where(tf.is_nan(tf.sqrt(self.predicted[:,:,4])), tf.zeros_like(self.predicted[:,:,4]), self.predicted[:,:,4]) - tf.sqrt(self.y[:,:,4]))
        #self.loss += tf.reduce_sum(tf.math.multiply(self.y[:,:,0], tmp1+tmp2))
        
        self.loss += tf.reduce_sum(tf.square(self.y[:,:,0] - self.predicted[:,:,0]))
        
        self.op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)
        
        self.sess = tf.Session() 
        self.reset()
    
    
    def NN(self, _in): 
        """
        Predicts the middle of the source and the dimensions of the box using a feed-forward net.
        Parameters: 
            arg1: 
                Values to pass into the net 
        Returns:    
            out1:
                Predicted output from the net. 
        """
        
        inp = tf.reshape(_in, (-1, (self.config.f + 2*self.config.c)**2))
        lay1 = tf.contrib.layers.fully_connected(inp, 128)
        drop1 = tf.nn.dropout(lay1, self.keep_prob)
        lay2= tf.contrib.layers.fully_connected(drop1, 128)
        drop2 = tf.nn.dropout(lay2, self.keep_prob)
        lay3 = tf.contrib.layers.fully_connected(drop2, 128, tf.nn.relu)
        drop3 = tf.nn.dropout(lay3, self.keep_prob)
        out = tf.contrib.layers.fully_connected(drop3, 5, \
                            activation_fn = tf.nn.relu)
        return out
    
    def CNN(self, _in): 
        """
        Predicts the middle of the source and the dimensions of the box.
        Uses convolution and max pool layers 
        Currently not used, testing needed to improve on feed-forward model
        Parameters: 
            arg1: 
                Values to pass into the net 
        Returns:    
            out1:
                Predicted output from the net. 
        """
        
        inp = tf.reshape(_in, (-1,self.config.f + 2*self.config.c,self.config.f + 2*self.config.c,1))
        conv1 = tf.contrib.layers.conv2d(inp,128, 
                                kernel_size = [3,3],
                                padding = 'same',
                                activation_fn = tf.nn.relu)
        pool1 = tf.contrib.layers.max_pool2d(conv1,
                                kernel_size = [2,2],
                                stride = 4)

        conv2 = tf.contrib.layers.conv2d(pool1,256,
                                kernel_size = [3,3],
                                padding = 'same',
                                activation_fn = tf.nn.relu)
        pool2 = tf.contrib.layers.max_pool2d(conv2,
                                kernel_size = [2,2],
                                stride = 4)
        pool2_flat = tf.reshape(pool2, [-1, 256])
    
        fc1 = tf.contrib.layers.fully_connected(pool2_flat, 1024,
                                activation_fn = tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 128,
                                activation_fn = tf.nn.relu)
        fc3 = tf.contrib.layers.fully_connected(fc2, 5,
                                activation_fn = tf.nn.relu)
        return fc3
    
    
    def reset(self):
        """
        resets the current tf session, and some history parameters
        """
        self.curr_epoch = 0
        self.loss_history = []
        self.sess.run(tf.global_variables_initializer()) 
        
    def train(self, inp, out, num_epoch): 
        """
        Trains this neural network on the input data, comparing to output values.
        Parameters:
            arg1: numpy.ndarray of size (___, config.L, config.L), dtype = np.float
                Input data from training set 
            arg2: numpy.ndarray of size (___, config.gridN**2, config.boxN*5), dtype = np.float
                Output data from the training set 
                arg1 and arg2 must have same size first dimension
            arg3: int
                How many passes of entire data set
        Returns:
            None
        """
        
        num_batches = int(inp.shape[0]/self.config.batch_size)
        inp, out = shuffle(inp,out)#ensures that each batch is representative of the data 
        
        for epoch in range(num_epoch): #how many times the net sees the entire data set 
            avg_loss = 0
            for batch in range(num_batches): #batch the data
                l, _ = self.sess.run([self.loss, self.op], feed_dict = {
                    self.x: inp[batch*self.config.batch_size:(batch+1)*self.config.batch_size],
                    self.y: out[batch*self.config.batch_size:(batch+1)*self.config.batch_size],
                    self.keep_prob: self.config.keep_prob})
                avg_loss += l
            if epoch%self.config.checkpoint == 0:
                print("Step " + str(self.curr_epoch) + ", Loss = " + "{:.4f}".format(avg_loss/num_batches))
            self.loss_history.append(avg_loss)
            self.curr_epoch += 1
    
    def test(self, inp, out): 
        """
        Gives loss for a test dataset, i.e. data the net has not seen during training
        Parameters:
            arg1: numpy.ndarray of size (___, config.L, config.L), dtype = np.float
                Input data from the test set
            arg2: numpy.ndarray of size (___, config.gridN**2, config.boxN*5), dtype = np.float
                Output data from the testing set
                arg1 and arg2 must have same size first dimension
        Returns:
            None
        """
        
        l = self.sess.run([self.loss], feed_dict = {
            self.x: inp, self.y: out, self.keep_prob: 1})      
        return l 

    def predict(self, inp): #gain prediction with no training 
        """
        Use the net for prediction
        Parameters:
            arg1: numpy.ndarray of size (___, config.L, config.L), dtype = np.float
                Input value for which you want to predict output
        Returns:
            arg2:
        """    
        
        preds = self.sess.run([self.predicted], feed_dict = {
            self.x: inp, self.keep_prob: 1})
        return preds[0]
    
    def save(self, filename = 'test_model'): 
        """
        Saves the current internal state for this network architecture
        """
        saver = tf.train.Saver()
        saver.save(self.sess, save_path='./checkpts/' + str(filename))
        
    def load(self, filename = 'test_model'):
        """
        Reloads a saved internal state of this network architecture
        """
        self.reset()
        saver = tf.train.Saver()
        saver.restore(self.sess, './checkpts/' + str(filename))
    

class EDNN_helper(object):
    """author: Kyle Mills"""
    def __init__(self,L,f,c):
        assert f <= L/2, "Focus must be less that half the image size to use this implementation."
        assert (f + 2*c) <= L, "Total tile size (f+2c) is larger than input image."
        self.l = L
        self.f = f
        self.c = c
        
    def __roll(self, in_, num, axis):
        D = tf.transpose(in_, perm=[axis,1-axis]) #if axis=1, transpose first
        D = tf.concat([D[num:,:], D[0:num, :]], axis=0)
        return tf.transpose(D, perm=[axis, 1-axis]) #if axis=1, transpose back

    def __slice(self, in_, x1, y1, w, h):
        return in_[x1:x1+w, y1:y1+h]

    def ednn_split(self,in_):
        tiles = []
        for iTile in range(int(self.l/self.f)):
            for jTile in range(int(self.l/self.f)):
                #calculate the indices of the centre of this tile (i.e. the centre of the focus region)
                cot = (iTile*self.f + self.f//2, jTile*self.f + self.f//2)
                foc_centered = in_
                #shift picture, wrapping the image around, 
                #so that focus is centered in the middle of the image
                foc_centered = self.__roll(foc_centered, int(self.l//2-cot[0]),0)
                foc_centered = self.__roll(foc_centered, int(self.l//2-cot[1]),1)
                #Finally slice away the excess image that we don't want to appear in this tile
                final = self.__slice(foc_centered, int(self.l//2-self.f//2-self.c),\
                    int(self.l//2-self.f//2-self.c), 2*self.c+self.f, 2*self.c+self.f)
                tiles.append(final)
        return tf.expand_dims(tiles, axis=3)
        
                
