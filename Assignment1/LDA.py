# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:11:06 2017

@author: Sherin
"""

import numpy as np
from matplotlib import pyplot
from scipy import linalg as LA
from scipy.spatial import distance
from sklearn import decomposition
from sklearn.decomposition import PCA
from numpy.linalg import inv
from sklearn.neighbors import NearestNeighbors


def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    pgmf.readline()
    #assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return raster

def flat_array(arr):
    raster = np.array(arr)
    flat = raster.flatten()
    return flat

def PCA1(d, alpha):
    mean1 = sum(d)
    n = d.shape[0]
    mean = mean1/n
    z = d-((np.transpose(mean)))
    B = np.reshape(mean, (-1, 92))
    pyplot.imshow(B, pyplot.cm.gray)
    pyplot.show()
    sigma = (1/n)*np.dot(np.transpose(z),z)
    e_vals, e_vecs = LA.eig(sigma)
    print(e_vecs)
    
    
    alphan = 0
    i=0
    all_sum = sum(e_vals)
    e_sum = 0
    
    while(alphan<alpha and i<e_vecs.shape[0]):
        alphan = e_sum/all_sum
        e_sum = e_sum + e_vals[i]
        print(alphan)
        i = i+1
    i = i-1
    print(i)
    u_vec = e_vecs[0:i:1]
    z_new = np.dot(d,np.transpose(u_vec))
    print(u_vec)
    return u_vec, z_new

def PCA2(d, alpha):

    
    z = d-d.mean(axis=0)
    
    pca = PCA(alpha)
    pca.fit(z)
    z_new = pca.transform(z)
    u = pca.components_
    print(u.shape)
    print(z.shape)
    print(z_new.shape)

    
    
    
    return u, z_new
def LDA (d,label):
    mean1 = sum(d)
    n = d.shape[0]
    mean = mean1/n
    mean_vectors=[]
    for cl in range(1,41):
     mean_vectors.append(np.mean(d[label==cl], axis=0))
     print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1])) 
    nk=10
    parameter= np.subtract(mean_vectors,mean)
    print(parameter.shape)
    parameterdottranspose=np.dot(parameter.transpose(),parameter)
    B=np.multiply(nk,parameterdottranspose)
    Z=[]
    S=np.zeros(10304)
    for cl in range(1,41):
     Z1=np.subtract(d[label==cl],mean_vectors[cl-1])
     Z.append(Z1)
     print('Z Center class matrices %s: %s\n' %(cl, Z[cl-1]))
     S1=np.dot(Z1.transpose(),Z1)
     print('S SCATTER class matrices %s: %s\n' %(cl,S1))
     S=S+S1
    print('the S MATRIX has values %s' %S)
    print(S.shape)
    invsS= inv(S)
    invSB=np.dot(invsS,B)
    print('the size of eign input s inverse and b')
    print(invSB.shape)
    print(invSB)
    w, v = LA.eigh(invSB,eigvals=(10304-39,10304-1))
    print('eign vectors')
    print(v.shape)
    return v
    
    
    


d = np.array([])
i=1
label = np.array([])
for i in range(1,41):
   for j in range(1,11): 
        label = np.hstack((label, i))
        


for i in range(1,41):
    for j in range(1,11):
        path ='D:/orl_faces/s' + str(i) + '/' + str(j) + '.pgm'
        f = open(path, 'rb')
        raster = read_pgm(f)
        flat = flat_array(raster)
        #pyplot.imshow(raster, pyplot.cm.gray)
        #pyplot.show()
        if i==1 and j == 1:
            d = flat
        else:
            d = np.vstack((d,flat))
            
        
        
trainingD = d[::2]
testingD = d[1::2]
trainingL = label[::2]
testingL = label[1::2]
U=LDA(trainingD,trainingL)
testingD_new = np.dot(testingD, U)
trainingD_new = np.dot(trainingD, U)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(trainingD_new)
distances, indices = nbrs.kneighbors(testingD_new)
print(indices)
tr = np.array([])
for j in range(0, indices.shape[0]):
   
    if(trainingL[indices[j]] == testingL[j]):
        tr = np.hstack((tr, 1))
    else:
        tr = np.hstack((tr, 0))  
print(tr)

accuracy = sum(tr)/(tr.shape)*100

print(accuracy)





