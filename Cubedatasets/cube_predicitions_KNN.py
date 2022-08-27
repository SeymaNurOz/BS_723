# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 18:18:43 2022

@author: seyma
"""
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math as m



 
def find_closer(v,dl,search_list):
    closer_index=0
    min_dist=m.sqrt((search_list[0][0]-v)**2+(search_list[0][1]-dl)**2)
    for i in range(len(search_list)):
        distance=m.sqrt((search_list[i][0]-v)**2+(search_list[i][1]-dl)**2)
        if distance<min_dist:
            min_dist=distance
            closer_index=i
    return closer_index


def find_closer_3d(x,y,z,search_list):
    closer_index=0
    min_dist=m.sqrt((search_list[0][0]-x)**2+(search_list[0][1]-y)**2+(search_list[0][2]-z)**2)
    for i in range(len(search_list)):
        distance=m.sqrt((search_list[i][0]-x)**2+(search_list[i][1]-y)**2+(search_list[i][2]-z)**2)
        if distance<min_dist:
            min_dist=distance
            closer_index=i
    return closer_index

"""CHALLENGE1"""

all_data=[[11520.0, 47.0, 255, 0, 255], [7920.0, 43.0, 255, 255, 0], [13728.0, 52.0, 255, 255, 0], [28000.0, 66.0, 0, 0, 255], [10200.0, 46.0, 0, 255, 0], [5184.0, 32.0, 255, 255, 255], [12800.0, 47.0, 255, 0, 255], [55296.0, 72.0, 0, 255, 0], [2240.0, 26.0, 255, 255, 255], [2992.0, 41.0, 0, 255, 255], [6720.0, 49.0, 255, 0, 0], [4032.0, 45.0, 255, 255, 0], [2592.0, 40.0, 255, 255, 0], [8400.0, 58.0, 0, 0, 255], [6336.0, 43.0, 0, 255, 255], [25024.0, 59.0, 0, 255, 0], [6480.0, 41.0, 255, 0, 255], [19712.0, 56.0, 0, 255, 0], [51984.0, 65.0, 0, 0, 0], [7488.0, 45.0, 0, 255, 255], [5376.0, 51.0, 0, 255, 255], [4800.0, 37.0, 255, 255, 0], [6240.0, 35.0, 255, 255, 255], [6688.0, 45.0, 255, 0, 255], [92000.0, 79.0, 0, 0, 0], [12096.0, 50.0, 255, 0, 255], [38080.0, 59.0, 0, 0, 0], [24840.0, 58.0, 255, 0, 0], [7392.0, 48.0, 0, 255, 255], [41184.0, 63.0, 0, 0, 255], [24840.0, 58.0, 0, 0, 255], [28800.0, 64.0, 0, 0, 255], [8064.0, 46.0, 0, 255, 0], [144.0, 9.0, 255, 255, 255], [36000.0, 65.0, 0, 255, 0], [85008.0, 76.0, 0, 0, 0], [44352.0, 63.0, 0, 0, 0], [5472.0, 45.0, 255, 0, 255], [44352.0, 63.0, 0, 0, 0], [7616.0, 45.0, 0, 255, 0], [4032.0, 45.0, 255, 255, 0], [24000.0, 60.0, 0, 255, 0], [24192.0, 58.0, 0, 255, 0], [18304.0, 47.0, 255, 255, 0], [29952.0, 60.0, 255, 0, 255], [25920.0, 53.0, 0, 0, 255], [2880.0, 41.0, 0, 255, 255], [18928.0, 46.0, 255, 255, 0], [7680.0, 35.0, 255, 255, 255], [16640.0, 46.0, 255, 255, 0], [36176.0, 58.0, 0, 0, 0], [3072.0, 35.0, 255, 0, 255], [6400.0, 45.0, 255, 255, 0], [13248.0, 59.0, 0, 0, 255], [28160.0, 56.0, 255, 0, 0], [10368.0, 45.0, 255, 255, 0], [5376.0, 51.0, 0, 255, 255], [7680.0, 47.0, 255, 255, 0], [5376.0, 51.0, 0, 255, 255], [3744.0, 33.0, 255, 255, 255], [39304.0, 59.0, 0, 0, 0], [9504.0, 57.0, 255, 0, 0], [6000.0, 37.0, 255, 255, 0], [17248.0, 54.0, 255, 0, 0], [512.0, 14.0, 255, 255, 255], [11520.0, 47.0, 255, 0, 255], [24840.0, 58.0, 0, 0, 255], [29376.0, 62.0, 255, 0, 0], [12000.0, 55.0, 0, 255, 255], [14784.0, 54.0, 0, 0, 255], [28160.0, 56.0, 0, 0, 255], [24192.0, 62.0, 255, 0, 0]]

volume=list(list(zip(*all_data))[0])
d_len=list(list(zip(*all_data))[1])
r_val=list(list(zip(*all_data))[2])
g_val=list(list(zip(*all_data))[3])
b_val=list(list(zip(*all_data))[4])

def div_255(num):
    return num/255


r2_val=list(map(div_255,r_val))
g2_val=list(map(div_255,g_val))
b2_val=list(map(div_255,b_val))
# 
vol_dl=list(map(list,zip(volume,d_len)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))
rgba_list=list(map(list,zip(r2_val,g2_val,b2_val)))
# print(rgb_list)
fig, ax = plt.subplots()
ax.scatter(x=volume,y=d_len,c=rgba_list)
ax.set_title('Volume/diagonal_length')
plt.show()


"""FIND CLOSER"""
X_train, X_validation, Y_train, Y_validation = train_test_split(vol_dl, rgb_list, test_size=0.25, random_state=2)

Y_pred=[]
for vol,d_l in X_validation:
    index=find_closer(vol,d_l,X_train)
    Y_pred.append(Y_train[index])
print("challenge1")
T=0
F=0
for i in range(len(Y_pred)):
    print(Y_pred[i],Y_validation[i])
    if Y_pred[i]==Y_validation[i]:
        T+=1
    else:F+=1



from sklearn.metrics import r2_score
print(T,F)
score=r2_score(Y_validation,Y_pred)
print(score)



"""CHALLENGE2"""

all_data=[[19360.0, 53.0, 89, 222, 100], [23760.0, 56.0, 78, 144, 222], [26112.0, 61.0, 67, 244, 166], [8064.0, 53.0, 155, 211, 11], [23920.0, 56.0, 122, 233, 89], [66240.0, 71.0, 233, 200, 177], [1920.0, 34.0, 33, 155, 11], [55296.0, 72.0, 244, 111, 244], [51072.0, 65.0, 155, 188, 211], [11440.0, 52.0, 33, 122, 222], [41472.0, 65.0, 244, 177, 111], [10120.0, 52.0, 100, 33, 233], [7296.0, 46.0, 22, 188, 111], [2560.0, 34.0, 22, 155, 33], [13728.0, 52.0, 122, 44, 222], [10560.0, 47.0, 44, 200, 100], [44352.0, 67.0, 244, 100, 211], [5760.0, 42.0, 22, 177, 89], [77616.0, 74.0, 211, 211, 222], [3264.0, 38.0, 67, 166, 11], [31616.0, 56.0, 122, 155, 188], [4320.0, 36.0, 22, 78, 144], [2016.0, 44.0, 211, 0, 44], [10400.0, 49.0, 200, 33, 122], [4752.0, 31.0, 44, 78, 100], [5016.0, 44.0, 188, 100, 11], [5376.0, 51.0, 55, 22, 244], [5280.0, 46.0, 200, 100, 11], [16744.0, 55.0, 233, 55, 122], [13984.0, 60.0, 22, 233, 188], [5280.0, 46.0, 11, 100, 200], [4320.0, 44.0, 200, 11, 78], [6480.0, 41.0, 78, 177, 33], [1200.0, 19.0, 33, 33, 44], [6688.0, 45.0, 100, 188, 22], [2400.0, 51.0, 255, 11, 22], [2016.0, 44.0, 44, 0, 211], [10400.0, 49.0, 200, 122, 33], [8624.0, 38.0, 55, 100, 133], [36800.0, 64.0, 200, 89, 233], [8800.0, 49.0, 89, 33, 222], [5984.0, 41.0, 22, 100, 166], [15000.0, 59.0, 255, 144, 33], [16128.0, 64.0, 211, 244, 22], [7168.0, 43.0, 133, 22, 155], [5376.0, 51.0, 55, 22, 244], [5016.0, 44.0, 11, 100, 188], [25200.0, 51.0, 133, 144, 144], [54264.0, 66.0, 166, 188, 211], [1760.0, 45.0, 222, 0, 33], [5600.0, 53.0, 255, 22, 55], [50688.0, 69.0, 244, 111, 222], [28800.0, 64.0, 67, 255, 177], [46368.0, 65.0, 133, 177, 233], [9856.0, 41.0, 55, 100, 155], [18816.0, 53.0, 67, 133, 211], [25920.0, 53.0, 111, 144, 177], [13104.0, 51.0, 122, 211, 44], [24480.0, 51.0, 111, 144, 166], [21280.0, 51.0, 188, 133, 89], [49248.0, 64.0, 177, 177, 188], [960.0, 24.0, 89, 44, 0], [2496.0, 31.0, 11, 67, 122], [10920.0, 50.0, 122, 211, 33], [512.0, 14.0, 22, 22, 22], [5040.0, 40.0, 55, 177, 33], [6048.0, 49.0, 11, 111, 211], [24576.0, 60.0, 244, 155, 67], [9504.0, 57.0, 177, 222, 11], [14112.0, 52.0, 133, 44, 211], [60000.0, 71.0, 255, 200, 144], [18768.0, 58.0, 44, 233, 166]]

volume=list(list(zip(*all_data))[0])
d_len=list(list(zip(*all_data))[1])
r_val=list(list(zip(*all_data))[2])
g_val=list(list(zip(*all_data))[3])
b_val=list(list(zip(*all_data))[4])

def div_255(num):
    return num/255


r2_val=list(map(div_255,r_val))
g2_val=list(map(div_255,g_val))
b2_val=list(map(div_255,b_val))
# 
vol_dl=list(map(list,zip(volume,d_len)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))
rgba_list=list(map(list,zip(r2_val,g2_val,b2_val)))
# print(rgb_list)
fig, ax = plt.subplots()
ax.scatter(x=volume,y=d_len,c=rgba_list)
ax.set_title('Volume/diagonal_length')
plt.show()




X_train, X_validation, Y_train, Y_validation = train_test_split(vol_dl, rgb_list, test_size=0.16, random_state=3)

Y_pred=[]
for vol,d_l in X_validation:
    index=find_closer(vol,d_l,X_train)
    Y_pred.append(Y_train[index])
print("challenge2")
T=0
for i in range(len(Y_pred)):
    print(Y_pred[i],Y_validation[i])
    if Y_pred[i]==Y_validation[i]:
        T+=1
    else:
        F+=1
print(T,F)
score=r2_score(Y_validation,Y_pred)
print(score)
    
    
"""CHALLENGE3"""

all_data=[[19360.0, 53.0, 89, 33, 100], [23760.0, 56.0, 78, 111, 222], [26112.0, 61.0, 67, 11, 166], [8064.0, 53.0, 155, 44, 11], [23920.0, 56.0, 122, 22, 89], [66240.0, 71.0, 233, 55, 177], [1920.0, 34.0, 33, 100, 11], [55296.0, 72.0, 244, 144, 244], [51072.0, 65.0, 155, 67, 211], [11440.0, 52.0, 33, 133, 222], [41472.0, 65.0, 244, 78, 111], [10120.0, 52.0, 100, 222, 233], [7296.0, 46.0, 22, 67, 111], [2560.0, 34.0, 22, 100, 33], [13728.0, 52.0, 122, 211, 222], [10560.0, 47.0, 44, 55, 100], [44352.0, 67.0, 244, 155, 211], [5760.0, 42.0, 22, 78, 89], [77616.0, 74.0, 211, 44, 222], [3264.0, 38.0, 67, 89, 11], [31616.0, 56.0, 122, 100, 188], [4320.0, 36.0, 22, 177, 144], [2016.0, 44.0, 211, 255, 44], [10400.0, 49.0, 200, 222, 122], [4752.0, 31.0, 44, 177, 100], [5016.0, 44.0, 188, 155, 11], [5376.0, 51.0, 55, 233, 244], [5280.0, 46.0, 200, 155, 11], [16744.0, 55.0, 233, 200, 122], [13984.0, 60.0, 22, 22, 188], [5280.0, 46.0, 11, 155, 200], [4320.0, 44.0, 200, 244, 78], [6480.0, 41.0, 78, 78, 33], [1200.0, 19.0, 33, 222, 44], [6688.0, 45.0, 100, 67, 22], [2400.0, 51.0, 255, 244, 22], [2016.0, 44.0, 44, 255, 211], [10400.0, 49.0, 200, 133, 33], [8624.0, 38.0, 55, 155, 133], [36800.0, 64.0, 200, 166, 233], [8800.0, 49.0, 89, 222, 222], [5984.0, 41.0, 22, 155, 166], [15000.0, 59.0, 255, 111, 33], [16128.0, 64.0, 211, 11, 22], [7168.0, 43.0, 133, 233, 155], [5376.0, 51.0, 55, 233, 244], [5016.0, 44.0, 11, 155, 188], [25200.0, 51.0, 133, 111, 144], [54264.0, 66.0, 166, 67, 211], [1760.0, 45.0, 222, 255, 33], [5600.0, 53.0, 255, 233, 55], [50688.0, 69.0, 244, 144, 222], [28800.0, 64.0, 67, 0, 177], [46368.0, 65.0, 133, 78, 233], [9856.0, 41.0, 55, 155, 155], [18816.0, 53.0, 67, 122, 211], [25920.0, 53.0, 111, 111, 177], [13104.0, 51.0, 122, 44, 44], [24480.0, 51.0, 111, 111, 166], [21280.0, 51.0, 188, 122, 89], [49248.0, 64.0, 177, 78, 188], [960.0, 24.0, 89, 211, 0], [2496.0, 31.0, 11, 188, 122], [10920.0, 50.0, 122, 44, 33], [512.0, 14.0, 22, 233, 22], [5040.0, 40.0, 55, 78, 33], [6048.0, 49.0, 11, 144, 211], [24576.0, 60.0, 244, 100, 67], [9504.0, 57.0, 177, 33, 11], [14112.0, 52.0, 133, 211, 211], [60000.0, 71.0, 255, 55, 144], [18768.0, 58.0, 44, 22, 166]]

volume=list(list(zip(*all_data))[0])
d_len=list(list(zip(*all_data))[1])
r_val=list(list(zip(*all_data))[2])
g_val=list(list(zip(*all_data))[3])
b_val=list(list(zip(*all_data))[4])

def div_255(num):
    return num/255

r2_val=list(map(div_255,r_val))
g2_val=list(map(div_255,g_val))
b2_val=list(map(div_255,b_val))
# 
vol_dl=list(map(list,zip(volume,d_len)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))
rgba_list=list(map(list,zip(r2_val,g2_val,b2_val)))
# print(rgb_list)
fig, ax = plt.subplots()
ax.scatter(x=volume,y=d_len,c=rgba_list)
ax.set_title('Volume/diagonal_length')
plt.show()




X_train, X_validation, Y_train, Y_validation = train_test_split(vol_dl, rgb_list, test_size=0.12, random_state=4)

Y_pred=[]
for vol,d_l in X_validation:
    index=find_closer(vol,d_l,X_train)
    Y_pred.append(Y_train[index])

T=0
print("challenge3")
for i in range(len(Y_pred)):
    print(Y_pred[i],Y_validation[i])
    if Y_pred[i]==Y_validation[i]:
        T+=1
score=r2_score(Y_validation,Y_pred)
print(score)


"""xyz ile"""

"""CH1"""
all_data=[[16, 40, 18, 255, 0, 255], [10, 22, 36, 255, 255, 0], [26, 12, 44, 255, 255, 0], [40, 50, 14, 0, 0, 255], [34, 10, 30, 0, 255, 0], [24, 18, 12, 255, 255, 255], [20, 40, 16, 255, 0, 255], [48, 24, 48, 0, 255, 0], [8, 14, 20, 255, 255, 255], [34, 4, 22, 0, 255, 255], [6, 40, 28, 255, 0, 0], [16, 6, 42, 255, 255, 0], [18, 4, 36, 255, 255, 0], [50, 28, 6, 0, 0, 255], [36, 22, 8, 0, 255, 255], [34, 16, 46, 0, 255, 0], [18, 36, 10, 255, 0, 255], [32, 14, 44, 0, 255, 0], [36, 38, 38, 0, 0, 0], [36, 8, 26, 0, 255, 255], [48, 8, 14, 0, 255, 255], [8, 20, 30, 255, 255, 0], [12, 20, 26, 255, 255, 255], [22, 38, 8, 255, 0, 255], [40, 46, 50, 0, 0, 0], [12, 42, 24, 255, 0, 255], [40, 34, 28, 0, 0, 0], [18, 46, 30, 255, 0, 0], [42, 8, 22, 0, 255, 255], [44, 36, 26, 0, 0, 255], [30, 46, 18, 0, 0, 255], [36, 50, 16, 0, 0, 255], [36, 8, 28, 0, 255, 0], [4, 6, 6, 255, 255, 255], [50, 20, 36, 0, 255, 0], [42, 44, 46, 0, 0, 0], [44, 36, 28, 0, 0, 0], [6, 38, 24, 255, 0, 255], [28, 36, 44, 0, 0, 0], [28, 8, 34, 0, 255, 0], [16, 6, 42, 255, 255, 0], [50, 16, 30, 0, 255, 0], [36, 16, 42, 0, 255, 0], [22, 26, 32, 255, 255, 0], [24, 48, 26, 255, 0, 255], [36, 30, 24, 0, 0, 255], [36, 4, 20, 0, 255, 255], [26, 26, 28, 255, 255, 0], [16, 20, 24, 255, 255, 255], [20, 26, 32, 255, 255, 0], [28, 34, 38, 0, 0, 0], [12, 32, 8, 255, 0, 255], [20, 8, 40, 255, 255, 0], [36, 46, 8, 0, 0, 255], [22, 32, 40, 255, 0, 0], [12, 24, 36, 255, 255, 0], [48, 8, 14, 0, 255, 255], [8, 24, 40, 255, 255, 0], [48, 8, 14, 0, 255, 255], [26, 18, 8, 255, 255, 255], [34, 34, 34, 0, 0, 0], [6, 44, 36, 255, 0, 0], [10, 20, 30, 255, 255, 0], [14, 28, 44, 255, 0, 0], [8, 8, 8, 255, 255, 255], [18, 40, 16, 255, 0, 255], [30, 46, 18, 0, 0, 255], [18, 48, 34, 255, 0, 0], [50, 12, 20, 0, 255, 255], [44, 28, 12, 0, 0, 255], [40, 32, 22, 0, 0, 255], [14, 48, 36, 255, 0, 0]]

x_lengths=list(list(zip(*all_data))[0])
y_lengths=list(list(zip(*all_data))[1])
z_lengths=list(list(zip(*all_data))[2])
r_val=list(list(zip(*all_data))[3])
g_val=list(list(zip(*all_data))[4])
b_val=list(list(zip(*all_data))[5])

dim_list=list(map(list,zip(x_lengths,y_lengths,z_lengths)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))

X_train, X_validation, Y_train, Y_validation = train_test_split(dim_list, rgb_list, test_size=0.25, random_state=578)

Y_pred=[]
for x,y,z in X_validation:
    index=find_closer_3d(x,y,z,X_train)
    Y_pred.append(Y_train[index])
    
results=[]
for i in range(len(Y_pred)):
    # print(Y_pred[i],Y_validation[i])
    interior=[]
    if Y_pred[i][0]==Y_validation[i][0]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][1]==Y_validation[i][1]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][2]==Y_validation[i][2]:
        interior.append("T")
    else:interior.append("F")
    results.append(interior)
    print(Y_pred[i],Y_validation[i],interior)

score=r2_score(Y_validation,Y_pred)
print(score)


"""ch2"""
all_data=[[20, 44, 22, 89, 222, 100], [18, 30, 44, 78, 144, 222], [16, 48, 34, 67, 244, 166], [32, 42, 6, 155, 211, 11], [26, 46, 20, 122, 233, 89], [46, 40, 36, 233, 200, 177], [10, 32, 6, 33, 155, 11], [48, 24, 48, 244, 111, 244], [32, 38, 42, 155, 188, 211], [10, 26, 44, 33, 122, 222], [48, 36, 24, 244, 177, 111], [22, 10, 46, 100, 33, 233], [8, 38, 24, 22, 188, 111], [8, 32, 10, 22, 155, 33], [26, 12, 44, 122, 44, 222], [12, 40, 22, 44, 200, 100], [48, 22, 42, 244, 100, 211], [8, 36, 20, 22, 177, 89], [42, 42, 44, 211, 211, 222], [16, 34, 6, 67, 166, 11], [26, 32, 38, 122, 155, 188], [8, 18, 30, 22, 78, 144], [42, 4, 12, 211, 0, 44], [40, 10, 26, 200, 33, 122], [12, 18, 22, 44, 78, 100], [38, 22, 6, 188, 100, 11], [14, 8, 48, 55, 22, 244], [40, 22, 6, 200, 100, 11], [46, 14, 26, 233, 55, 122], [8, 46, 38, 22, 233, 188], [6, 22, 40, 11, 100, 200], [40, 6, 18, 200, 11, 78], [18, 36, 10, 78, 177, 33], [10, 10, 12, 33, 33, 44], [22, 38, 8, 100, 188, 22], [50, 6, 8, 255, 11, 22], [12, 4, 42, 44, 0, 211], [40, 26, 10, 200, 122, 33], [14, 22, 28, 55, 100, 133], [40, 20, 46, 200, 89, 233], [20, 10, 44, 89, 33, 222], [8, 22, 34, 22, 100, 166], [50, 30, 10, 255, 144, 33], [42, 48, 8, 211, 244, 22], [28, 8, 32, 133, 22, 155], [14, 8, 48, 55, 22, 244], [6, 22, 38, 11, 100, 188], [28, 30, 30, 133, 144, 144], [34, 38, 42, 166, 188, 211], [44, 4, 10, 222, 0, 33], [50, 8, 14, 255, 22, 55], [48, 24, 44, 244, 111, 222], [16, 50, 36, 67, 255, 177], [28, 36, 46, 133, 177, 233], [14, 22, 32, 55, 100, 155], [16, 28, 42, 67, 133, 211], [24, 30, 36, 111, 144, 177], [26, 42, 12, 122, 211, 44], [24, 30, 34, 111, 144, 166], [38, 28, 20, 188, 133, 89], [36, 36, 38, 177, 177, 188], [20, 12, 4, 89, 44, 0], [6, 16, 26, 11, 67, 122], [26, 42, 10, 122, 211, 33], [8, 8, 8, 22, 22, 22], [14, 36, 10, 55, 177, 33], [6, 24, 42, 11, 111, 211], [48, 32, 16, 244, 155, 67], [36, 44, 6, 177, 222, 11], [28, 12, 42, 133, 44, 211], [50, 40, 30, 255, 200, 144], [12, 46, 34, 44, 233, 166]]

x_lengths=list(list(zip(*all_data))[0])
y_lengths=list(list(zip(*all_data))[1])
z_lengths=list(list(zip(*all_data))[2])
r_val=list(list(zip(*all_data))[3])
g_val=list(list(zip(*all_data))[4])
b_val=list(list(zip(*all_data))[5])

dim_list=list(map(list,zip(x_lengths,y_lengths,z_lengths)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))

X_train, X_validation, Y_train, Y_validation = train_test_split(dim_list, rgb_list, test_size=0.16, random_state=6)

Y_pred=[]
for x,y,z in X_validation:
    index=find_closer_3d(x,y,z,X_train)
    Y_pred.append(Y_train[index])
    
    results=[]
for i in range(len(Y_pred)):

    interior=[]
    if Y_pred[i][0]>0.8*Y_validation[i][0] and Y_pred[i][0]<1.2*Y_validation[i][0]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][1]>0.8*Y_validation[i][1] and Y_pred[i][1]<1.2*Y_validation[i][1]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][2]>0.8*Y_validation[i][2] and Y_pred[i][2]<1.2*Y_validation[i][2]:
        interior.append("T")
    else:interior.append("F")
    results.append(interior)
    print(Y_pred[i],Y_validation[i],interior)


score=r2_score(Y_validation,Y_pred)
print(score)



"""ch3"""
all_data=[[20, 44, 22, 89, 33, 100], [18, 30, 44, 78, 111, 222], [16, 48, 34, 67, 11, 166], [32, 42, 6, 155, 44, 11], [26, 46, 20, 122, 22, 89], [46, 40, 36, 233, 55, 177], [10, 32, 6, 33, 100, 11], [48, 24, 48, 244, 144, 244], [32, 38, 42, 155, 67, 211], [10, 26, 44, 33, 133, 222], [48, 36, 24, 244, 78, 111], [22, 10, 46, 100, 222, 233], [8, 38, 24, 22, 67, 111], [8, 32, 10, 22, 100, 33], [26, 12, 44, 122, 211, 222], [12, 40, 22, 44, 55, 100], [48, 22, 42, 244, 155, 211], [8, 36, 20, 22, 78, 89], [42, 42, 44, 211, 44, 222], [16, 34, 6, 67, 89, 11], [26, 32, 38, 122, 100, 188], [8, 18, 30, 22, 177, 144], [42, 4, 12, 211, 255, 44], [40, 10, 26, 200, 222, 122], [12, 18, 22, 44, 177, 100], [38, 22, 6, 188, 155, 11], [14, 8, 48, 55, 233, 244], [40, 22, 6, 200, 155, 11], [46, 14, 26, 233, 200, 122], [8, 46, 38, 22, 22, 188], [6, 22, 40, 11, 155, 200], [40, 6, 18, 200, 244, 78], [18, 36, 10, 78, 78, 33], [10, 10, 12, 33, 222, 44], [22, 38, 8, 100, 67, 22], [50, 6, 8, 255, 244, 22], [12, 4, 42, 44, 255, 211], [40, 26, 10, 200, 133, 33], [14, 22, 28, 55, 155, 133], [40, 20, 46, 200, 166, 233], [20, 10, 44, 89, 222, 222], [8, 22, 34, 22, 155, 166], [50, 30, 10, 255, 111, 33], [42, 48, 8, 211, 11, 22], [28, 8, 32, 133, 233, 155], [14, 8, 48, 55, 233, 244], [6, 22, 38, 11, 155, 188], [28, 30, 30, 133, 111, 144], [34, 38, 42, 166, 67, 211], [44, 4, 10, 222, 255, 33], [50, 8, 14, 255, 233, 55], [48, 24, 44, 244, 144, 222], [16, 50, 36, 67, 0, 177], [28, 36, 46, 133, 78, 233], [14, 22, 32, 55, 155, 155], [16, 28, 42, 67, 122, 211], [24, 30, 36, 111, 111, 177], [26, 42, 12, 122, 44, 44], [24, 30, 34, 111, 111, 166], [38, 28, 20, 188, 122, 89], [36, 36, 38, 177, 78, 188], [20, 12, 4, 89, 211, 0], [6, 16, 26, 11, 188, 122], [26, 42, 10, 122, 44, 33], [8, 8, 8, 22, 233, 22], [14, 36, 10, 55, 78, 33], [6, 24, 42, 11, 144, 211], [48, 32, 16, 244, 100, 67], [36, 44, 6, 177, 33, 11], [28, 12, 42, 133, 211, 211], [50, 40, 30, 255, 55, 144], [12, 46, 34, 44, 22, 166]]

x_lengths=list(list(zip(*all_data))[0])
y_lengths=list(list(zip(*all_data))[1])
z_lengths=list(list(zip(*all_data))[2])
r_val=list(list(zip(*all_data))[3])
g_val=list(list(zip(*all_data))[4])
b_val=list(list(zip(*all_data))[5])

dim_list=list(map(list,zip(x_lengths,y_lengths,z_lengths)))
rgb_list=list(map(list,zip(r_val,g_val,b_val)))

X_train, X_validation, Y_train, Y_validation = train_test_split(dim_list, rgb_list, test_size=0.12, random_state=6)

Y_pred=[]
for x,y,z in X_validation:
    index=find_closer_3d(x,y,z,X_train)
    Y_pred.append(Y_train[index])
    
    results=[]
for i in range(len(Y_pred)):

    interior=[]
    if Y_pred[i][0]>0.9*Y_validation[i][0] and Y_pred[i][0]<1.1*Y_validation[i][0]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][1]>0.9*Y_validation[i][1] and Y_pred[i][1]<1.1*Y_validation[i][1]:
        interior.append("T")
    else:interior.append("F")
    if Y_pred[i][2]>0.9*Y_validation[i][2] and Y_pred[i][2]<1.1*Y_validation[i][2]:
        interior.append("T")
    else:interior.append("F")
    results.append(interior)
    print(Y_pred[i],Y_validation[i],interior)
    

            
score=r2_score(Y_validation,Y_pred)
print(score)












