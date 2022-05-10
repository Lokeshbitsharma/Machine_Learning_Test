# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:47:05 2022

@author: lokes
"""

import numpy as np
import pandas as pd

import tensorflow as tf


class ZOO:
    
    def __init__(self,
               input_shape,
               prediction_service,
               decision_threshold,
               initial_const,
               confidence,
               learning_rate,
               box_max,
               box_min,
               max_iterations):
        self.input_shape = input_shape
        self.prediction_service = prediction_service
        self.model = prediction_service.model
        self.decision_threshold = decision_threshold
        
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
                
        self.const = tf.ones(input_shape) * initial_const
        
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        self.box_max = box_max 
        self.box_min = box_min
        self.boxmul = (self.box_max - self.box_min) / 2.
        self.boxplus = (self.box_min + self.box_max) / 2.
        
    def attack(self,x):
        self.modifier = tf.Variable(tf.zeros(self.input_shape, dtype=x.dtype), trainable=True)
        y = self.prediction_service.get_predictions(x.numpy(), p = self.decision_threshold, is_Transformed = True)
        
        prev = None
        iteration_count = 0
        for iteration in range(self.max_iterations):
            # print(iteration)
            x_new, loss, preds, l2_dist = self.attack_step(x, y, self.modifier)
            
            # if self.max_iterations%10 == 0:
            #     if prev is not None:
            #         if loss > prev * 0.9999:
            #             break  
            
            if prev is not None:
                if loss > prev * 0.999:
                    break 
            prev = loss
            iteration_count = iteration
        return x_new,loss,iteration_count
            
        
    def attack_step(self,x, y, modifier):
        x_new, grads, loss, preds, l2_dist = self.gradient(x, y, modifier)    
        self.optimizer.apply_gradients([(grads, modifier)])
        return x_new, loss, preds, l2_dist 
        
    def gradient(self, x, y, modifier):
        with tf.GradientTape() as tape:
            adv_x =  modifier + x
            x_new = self.clip_tanh(adv_x) 
            # print(x_new)
            preds = self.model(x_new)
            
            # y_true = self.get_y_true(y)
            # y_pred = self.get_y_pred(preds)
            y_pred = preds[0][0]
                   
            loss, l2_dist = self.loss_fn(
                x=x,
                x_new=x_new,
                y_true= y[0][0],
                y_pred = y_pred)
    
        grads = tape.gradient(loss, x_new)
        
        return x_new, grads, loss, preds, l2_dist
        
    def loss_fn(self, x,
                x_new,
                y_true,
                y_pred):
        
        # L2- norm         
        l2_dist = self.l2(x_new, x)
        
        # real = tf.reduce_sum(y_true * y_pred)    
        real = tf.reduce_max(y_pred)
        
        # c* f(x)
        if y_true == 1:
            loss_1 = tf.maximum(- self.confidence, real - self.decision_threshold)
        else:
            loss_1 = tf.maximum(- self.confidence, -real + self.decision_threshold)
            
        # sum up losses
        loss_2 = tf.reduce_sum(l2_dist)
        loss_1 = tf.reduce_sum(self.const * loss_1)
        loss = loss_1 + loss_2
        
        # print('L2_norm: {}, Loss_1: {}, Total Loss: {}'.format(loss_2, loss_1, loss))
        
        return loss, l2_dist
   
    def l2(self, x, y):
        # technically squarred l2
        return tf.reduce_sum(tf.square(x - y), list(range(1, len(self.input_shape))))

        
    def clip_tanh(self,input_x):
        clipped_input = tf.tanh(input_x) * self.boxmul + self.boxplus
        return clipped_input
    
    def get_y_pred(self, y_pred):
        p = y_pred[0][0]
        return tf.convert_to_tensor(np.array([p,1-p]), dtype= tf.float32)
    
    def get_y_true(self, y):
        if y[0][0] == 0:
            return tf.convert_to_tensor(np.array([0,1]), dtype= tf.float32)
        return tf.convert_to_tensor(np.array([1,0]), dtype= tf.float32)