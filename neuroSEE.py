# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:40:34 2022

@author: arsma
"""


import numpy as np
from skimage import io
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def vidProcessing(Video):
    m=Video.shape[1]
    n=Video.shape[2]
    maxI=np.empty([m, n], dtype=np.single)
    minI=np.empty([m, n], dtype=np.single)
    for i in range(m):
        for j in range(n):
            v=Video[:,i,j]
            maxI[i][j]=max(v)
            minI[i][j]=min(v)
    return maxI, minI

def imgProcessing(maxI, minI, dim):
    maxI=cv2.resize(maxI, dim)
    maxIN=(maxI-np.amin(maxI))/(np.amax(maxI))
    minI=cv2.resize(minI, dim)
    minIN=(minI-np.amin(minI))/(np.amax(minI))
    return maxI, minI, maxIN, minIN

def Cutter(folder, Arr, m=126, n=126, d=9, s=3):
    c=0
    for i in range(0, m-d+1, s):
        for j in range(0, n-d+1, s):
            cv2.imwrite(folder + f"//{c} {i} {j}.jpg", Arr[i:i+d,j:j+d])
            c+=1
    
def prepareLabels(path, folder):
    dim=(504,504)
    img=cv2.imread(path)
    img=cv2.resize(img, dim)
    if not os.path.exists(folder):   os.mkdir(folder)
    Cutter(folder, img, dim[0], dim[1], 36, 12)#504 126
    
def Cut(Mini, Maxi, m=126, n=126, d=9, s=3):
    l=[]
    h=((m-d)/s)+1
    h=int(h*h)
    MiniMax=np.empty([h,d,d,2], dtype=np.single)
    MiniMaxFlat=np.empty([h,2,d*d], dtype=np.single)
    c=0
    for i in range(0, m-d+1, s):
        for j in range(0 ,n-d+1, s):
            MiniMax[c,:,:,0]=Mini[i:i+d,j:j+d]-0.5
            MiniMaxFlat[c,0,:]=Mini[i:i+d,j:j+d].flatten()-0.5
            MiniMaxFlat[c,1,:]=Maxi[i:i+d,j:j+d].flatten()-0.5
            MiniMax[c,:,:,1]=Maxi[i:i+d,j:j+d]-0.5
            l.append((i,j))
            c+=1
    return MiniMax, MiniMaxFlat, l, h

def readLabels(path,st):
    y=[0]*st      
    for image in os.listdir("C://Users/arsma/OneDrive/Desktop/New folder (3)"):
        y[int(image.split()[0])]=1        
    return np.array(y)


def graphing(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    ax = plt.gca()
    ax.set_ylim([0.4,1])
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    ax = plt.gca()
    ax.set_ylim([0,1.8])
    plt.show()
 
    
def model_1d(XF,y):
    
    model=models.Sequential()
    model.add(Bidirectional(layers.LSTM(24, activation='tanh'), input_shape=(2,81)))
    model.add(layers.Dense(2, activation = 'softmax', kernel_regularizer =tf.keras.regularizers.l2(0.3)))
    model.summary()
    
    model.compile(optimizer= 'adam', loss=tf.keras.losses.categorical_crossentropy
                  , metrics= [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])
    
    X_train, X_CV, y_train, y_CV = train_test_split(XF, y, test_size=0.2)
    history = model.fit(X_train,to_categorical(y_train),epochs=990,batch_size=32,validation_data=(X_CV,to_categorical(y_CV)))
    model.save('model_1d.h5')
    
    graphing(history)
    
    
def model_2d(X,y):
    
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'tanh', input_shape = (9,9,2)))
    model.add(layers.MaxPooling2D(pool_size = [2,2]))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = [2,2]))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation = 'sigmoid'))
    model.add(layers.Dense(2, activation = 'softmax',kernel_regularizer =tf.keras.regularizers.l2(0.3)))
    model.summary()
    
    model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy
                  , metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])
    
    X_train, X_CV, y_train, y_CV = train_test_split(X, y, test_size=0.2)
    history = model.fit(X_train,to_categorical(y_train),epochs=330,batch_size=20,validation_data=(X_CV, to_categorical(y_CV)))
    model.save('model_2d.h5')
    
    graphing(history)


def model_3d(X,y):    
    
    model = models.Sequential()
    model.add(layers.Conv3D(32, kernel_size = [3,3,1], activation = 'tanh', padding = 'same', input_shape = (9,9,2,1)))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Conv3D(64, kernel_size = [3,3,2], activation = 'relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation = 'sigmoid'))
    model.add(layers.Dense(2, activation = 'softmax', kernel_regularizer =tf.keras.regularizers.l2(0.03)))
    model.summary()
    
    model.compile(optimizer = 'adam', loss = tf.keras.losses.categorical_crossentropy, metrics = [tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),"accuracy"])
    
    X = np.expand_dims(X, axis=4)
    X_train, X_CV, y_train, y_CV = train_test_split(X, y, test_size=0.2)
    history = model.fit(X_train,to_categorical(y_train),epochs=300,batch_size=32,validation_data=(X_CV,to_categorical(y_CV)))
    model.save('model_3d.h5')
    
    graphing(history)

    
def predictor(model,X):
    p=model.predict(X)
    yp=p[:,1]>p[:,0]
    return yp
    
def predictors(X,XF,t=False):
    model=models.load_model('model_1d.h5')
    yp=predictor(model,XF)
    modell=models.load_model('model_2d.h5')
    ypp=predictor(modell,X)
    modelll=models.load_model('model_3d.h5')
    yppp=predictor(modelll,np.expand_dims(X, axis=4))
    voting=yp+1-1
    voting=(voting+ypp)+yppp
    votBool=voting>1
    if t:   return yp,ypp,yppp,votBool,voting
    return votBool
    
def testing(X,XF,y):
    l=len(y)
    yp,ypp,yppp,votBool,voting= predictors(X,XF,True)
    print(f"The accuracy of 1d predictor = {np.mean(y==yp)} with {np.sum(y!=yp)} errors out of {l} .")
    print(f"The accuracy of 2d predictor = {np.mean(y==ypp)} with {np.sum(y!=ypp)} errors out of {l} .")
    print(f"The accuracy of 3d predictor = {np.mean(y==yppp)} with {np.sum(y!=yppp)} errors out of {l} .")
    print(f"The accuracy of  homoPredictor = {np.mean(y==votBool)} with {np.sum(y!=votBool)} errors out of {l} .")
    print("The testing errors on voting are :")
    for i in range(l): 
        if y[i]!=votBool[i]:    
            print(i,"   ",votBool[i],"   ",voting[i])    
    
def neuroSearch(votBool,loc,maxI):
    grid={}
    for i in range(len(votBool)):
        a, b = loc[i]
        if votBool[i]:  grid[loc[i]]=np.mean(maxI[a:a+9,b:b+9])
    gridV=list(grid.keys())
    neurons=[]
    for g in gridV:
        a=(g[0]+3,g[1])
        b=(g[0],g[1]+3)
        c=(g[0]+3,g[1]+3)
        d=(g[0]+6,g[1]+6)
        maxi=g
        if a in gridV:  
            if grid[a]>grid[maxi]:  maxi=a
            gridV.remove(a)
        if b in gridV:  
            if grid[b]>grid[maxi]:  maxi=b
            gridV.remove(b)
        if c in gridV:  
            if grid[c]>grid[maxi]:  maxi=c
            if not d in gridV:  gridV.remove(c)
        neurons.append(maxi)
    return neurons   

def neuroArt(img,neurons,m=1):
    # Create figure and axes
    fig, ax = plt.subplots()
     
    # Display the image
    ax.imshow(img)
    
    # Create a Rectangle patch
    # Add the patch to the Axes
    
    for x in neurons:
        rect = patches.Rectangle((int(x[1]*m), int(x[0]*m)), 
            int(9*m), int(9*m), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
    plt.show()    
    
def neuroIntensity(Video,neurons,dim):
    avgIntensity=np.empty([Video.shape[0],len(neurons)])
    i=0
    for img in Video:
        img=cv2.resize(img,dim)
        avgIntensity[i,:]=[np.mean(img[x[0]:x[0]+9,x[1]:x[1]+9]) for x in neurons]
        i+=1
    return avgIntensity
    
def apply(path):
    Video = io.imread(path)
    maxI, minI = vidProcessing(Video)
    dim=(126,126)
    maxI, minI, maxIN, minIN = imgProcessing(maxI, minI, dim)
    prepareLabels("Captured.PNG", "neuroNumber9")
    # putting manual labels on the images 
    X, XF, loc, st = Cut(minIN, maxIN)    
    y = readLabels("C://Users/arsma/OneDrive/Desktop/New folder (3)", st)
    model_1d(XF,y)
    model_2d(X,y)
    model_3d(X,y)
    testing(X,XF,y)
    yp=predictors(X,XF)
    neurons=neuroSearch(yp,loc,maxI)
    neuroArt(minI,neurons)
    neuroArt(maxI,neurons)
    neuroArt(cv2.resize(cv2.imread("Captured.PNG"),dim),neurons)
    
def application(path):
    Video = io.imread(path)
    maxI, minI = vidProcessing(Video)
    #maxI=cv2.resize(maxI,(945,945))
    #minI=cv2.resize(minI,(945,945))
    Neurons=[]
    pNeurons=[]
    Dim=[(i*126,i*126) for i in range(1,9)]
    for dim in Dim:
        a, b, maxIN, minIN = imgProcessing(maxI, minI, dim)
        X, XF, loc, st = Cut(minIN, maxIN, dim[0], dim[1])    
        yp=predictors(X,XF)
        Neurons.append(neuroSearch(yp,loc,a))
        pNeurons.append(len(Neurons[-1])/len(yp))    
    i=np.argmax(pNeurons)    
    neurons=Neurons[i]
    avgIntensity=neuroIntensity(Video,neurons,Dim[i])
    mult=maxI.shape[0]/Dim[i][0]
    neuroArt(minI,neurons,mult)
    neuroArt(maxI,neurons,mult)
    #neuroArt(cv2.resize(cv2.imread("Captured.PNG"),maxI.shape),neurons,mult)#  
    return avgIntensity

#apply('raw128.tif')    
# avgIntensity=application('raw128.tif')


'''
def applied(path):
    Video = io.imread(path)
    maxI, minI = vidProcessing(Video)
    
    maxI=maxI.transpose()#
    minI=minI.transpose()#
    
    dim=(126,126)
    maxI, minI, maxIN, minIN = imgProcessing(maxI, minI, dim)
    X, XF, loc, st = Cut(minIN, maxIN, dim[0], dim[1])    
    
    y = readLabels("C://Users/arsma/OneDrive/Desktop/New folder (3)", st)#
    y=((y.reshape(40,40)).transpose()).flatten()#
    testing(X,XF,y)#
    
    yp=predictors(X,XF)
    neurons=neuroSearch(yp,loc,maxI)
    #print(len(neurons)/len(yp))
    neuroArt(minI,neurons)
    neuroArt(maxI,neurons)
    neuroArt(cv2.resize(cv2.imread("Captured.PNG").transpose(1,0,2),dim),neurons)
    
def appliedPro(path):
    Video = io.imread(path)
    maxI, minI = vidProcessing(Video)
    maxI=cv2.resize(cv2.flip(maxI,2),(600,600))#
    minI=cv2.resize(cv2.flip(minI,2),(600,600))#
    Neurons=[]
    pNeurons=[]
    Dim=[(i*126,i*126) for i in range(1,9)]
    for dim in Dim:
        a, b, maxIN, minIN = imgProcessing(maxI, minI, dim)
        X, XF, loc, st = Cut(minIN, maxIN, dim[0], dim[1])    
        if dim==(126,126):
             y = readLabels("C://Users/arsma/OneDrive/Desktop/New folder (3)", st)#
             y=cv2.flip((y.reshape(40,40)),2).flatten()#
             testing(X,XF,y)#
        yp=predictors(X,XF)
        Neurons.append(neuroSearch(yp,loc,a))
        pNeurons.append(len(Neurons[-1])/len(yp))    
    i=np.argmax(pNeurons)    
    neurons=Neurons[i]
    mult=maxI.shape[0]/Dim[i][0]
    neuroArt(minI,neurons,mult)
    neuroArt(maxI,neurons,mult)
    neuroArt(cv2.resize(cv2.flip(cv2.imread("Captured.PNG"),2),maxI.shape),neurons,mult)  
    
#applied('raw128.tif')
#appliedPro('raw128.tif')   
'''