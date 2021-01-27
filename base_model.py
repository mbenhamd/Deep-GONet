import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class BaseModel():

    def __init__(self,X,n_input,n_classes,n_hidden_1,n_hidden_2,n_hidden_3,n_hidden_4,n_hidden_5,n_hidden_6,is_training,keep_prob):
        
        
        # Parameters
        self.X = X
        self.n_input = n_input
        self.n_classes=n_classes
        self.n_hidden_1=n_hidden_1
        self.n_hidden_2=n_hidden_2
        self.n_hidden_3=n_hidden_3
        self.n_hidden_4=n_hidden_4
        self.n_hidden_5=n_hidden_5
        self.n_hidden_6=n_hidden_6

        # Hyperparameters
        self.keep_prob = keep_prob # Dropout
        self.is_training = is_training # BN


    def store_layer_weights_and_bias(self):
        self.weights = {
            'h1_w': tf.get_variable('W1', shape=(self.n_input, self.n_hidden_1), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h2_w': tf.get_variable('W2', shape=(self.n_hidden_1, self.n_hidden_2), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h3_w': tf.get_variable('W3', shape=(self.n_hidden_2, self.n_hidden_3), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h4_w': tf.get_variable('W4', shape=(self.n_hidden_3, self.n_hidden_4), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h5_w': tf.get_variable('W5', shape=(self.n_hidden_4, self.n_hidden_5), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h6_w': tf.get_variable('W6', shape=(self.n_hidden_5, self.n_hidden_6), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'out_w': tf.get_variable('W_out',shape=(self.n_hidden_6, self.n_classes), initializer=tf.contrib.layers.variance_scaling_initializer())
        }
        self.biases = {
            'h1_b': tf.get_variable('B1',shape=(self.n_hidden_1),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h2_b': tf.get_variable('B2',shape=(self.n_hidden_2),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h3_b': tf.get_variable('B3',shape=(self.n_hidden_3),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h4_b': tf.get_variable('B4',shape=(self.n_hidden_4),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h5_b': tf.get_variable('B5',shape=(self.n_hidden_5),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'h6_b': tf.get_variable('B6',shape=(self.n_hidden_6),initializer=tf.contrib.layers.variance_scaling_initializer()),
            'out_b': tf.get_variable('B_out',shape=(self.n_classes),initializer=tf.contrib.layers.variance_scaling_initializer())
        }

    def fc(self,input,weights,biases,name,dim):
        h = tf.add(tf.matmul(input, weights), biases)
        if FLAGS.bn:
            h = tf.layers.batch_normalization(h,training=self.is_training,name='bn_'+name)
        h = tf.nn.relu(h, name=name)
        h = tf.nn.dropout(h, self.keep_prob)
        return h

    def net(self):  
        self.h1 = self.fc(self.X,self.weights['h1_w'],self.biases['h1_b'],name='layer1',dim=self.n_hidden_1)
        self.h2 = self.fc(self.h1,self.weights['h2_w'],self.biases['h2_b'],name='layer2',dim=self.n_hidden_2)
        self.h3 = self.fc(self.h2, self.weights['h3_w'], self.biases['h3_b'],name='layer3',dim=self.n_hidden_3)
        self.h4 = self.fc(self.h3,self.weights['h4_w'],self.biases['h4_b'],name='layer4',dim=self.n_hidden_4)
        self.h5 = self.fc(self.h4,self.weights['h5_w'],self.biases['h5_b'],name='layer5',dim=self.n_hidden_5)
        self.h6 = self.fc(self.h5, self.weights['h6_w'], self.biases['h6_b'],name='layer6',dim=self.n_hidden_6)
        output_layer = tf.add(tf.matmul(self.h6, self.weights['out_w']), self.biases['out_b'],name='output')     
        return output_layer

    def __call__(self):
        self.store_layer_weights_and_bias()
        return self.net()
        
