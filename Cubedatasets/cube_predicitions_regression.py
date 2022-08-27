# -*- coding: utf-8 -*-
"""
Created on Thu May  5 22:53:35 2022

@author: seyma
"""

# make predictions
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


"""CHALLENGE 1"""
all_data=[[16, 40, 18, 255, 0, 255], [10, 22, 36, 255, 255, 0], [26, 12, 44, 255, 255, 0], [40, 50, 14, 0, 0, 255], [34, 10, 30, 0, 255, 0], [24, 18, 12, 255, 255, 255], [20, 40, 16, 255, 0, 255], [48, 24, 48, 0, 255, 0], [8, 14, 20, 255, 255, 255], [34, 4, 22, 0, 255, 255], [6, 40, 28, 255, 0, 0], [16, 6, 42, 255, 255, 0], [18, 4, 36, 255, 255, 0], [50, 28, 6, 0, 0, 255], [36, 22, 8, 0, 255, 255], [34, 16, 46, 0, 255, 0], [18, 36, 10, 255, 0, 255], [32, 14, 44, 0, 255, 0], [36, 38, 38, 0, 0, 0], [36, 8, 26, 0, 255, 255], [48, 8, 14, 0, 255, 255], [8, 20, 30, 255, 255, 0], [12, 20, 26, 255, 255, 255], [22, 38, 8, 255, 0, 255], [40, 46, 50, 0, 0, 0], [12, 42, 24, 255, 0, 255], [40, 34, 28, 0, 0, 0], [18, 46, 30, 255, 0, 0], [42, 8, 22, 0, 255, 255], [44, 36, 26, 0, 0, 255], [30, 46, 18, 0, 0, 255], [36, 50, 16, 0, 0, 255], [36, 8, 28, 0, 255, 0], [4, 6, 6, 255, 255, 255], [50, 20, 36, 0, 255, 0], [42, 44, 46, 0, 0, 0], [44, 36, 28, 0, 0, 0], [6, 38, 24, 255, 0, 255], [28, 36, 44, 0, 0, 0], [28, 8, 34, 0, 255, 0], [16, 6, 42, 255, 255, 0], [50, 16, 30, 0, 255, 0], [36, 16, 42, 0, 255, 0], [22, 26, 32, 255, 255, 0], [24, 48, 26, 255, 0, 255], [36, 30, 24, 0, 0, 255], [36, 4, 20, 0, 255, 255], [26, 26, 28, 255, 255, 0], [16, 20, 24, 255, 255, 255], [20, 26, 32, 255, 255, 0], [28, 34, 38, 0, 0, 0], [12, 32, 8, 255, 0, 255], [20, 8, 40, 255, 255, 0], [36, 46, 8, 0, 0, 255], [22, 32, 40, 255, 0, 0], [12, 24, 36, 255, 255, 0], [48, 8, 14, 0, 255, 255], [8, 24, 40, 255, 255, 0], [48, 8, 14, 0, 255, 255], [26, 18, 8, 255, 255, 255], [34, 34, 34, 0, 0, 0], [6, 44, 36, 255, 0, 0], [10, 20, 30, 255, 255, 0], [14, 28, 44, 255, 0, 0], [8, 8, 8, 255, 255, 255], [18, 40, 16, 255, 0, 255], [30, 46, 18, 0, 0, 255], [18, 48, 34, 255, 0, 0], [50, 12, 20, 0, 255, 255], [44, 28, 12, 0, 0, 255], [40, 32, 22, 0, 0, 255], [14, 48, 36, 255, 0, 0]]
# all_data=[[16, 40, 18, '(255, 0, 255)'], [10, 22, 36, '(255, 255, 0)'], [26, 12, 44, '(255, 255, 0)'], [40, 50, 14, '(0, 0, 255)'], [34, 10, 30, '(0, 255, 0)'], [24, 18, 12, '(255, 255, 255)'], [20, 40, 16, '(255, 0, 255)'], [48, 24, 48, '(0, 255, 0)'], [8, 14, 20, '(255, 255, 255)'], [34, 4, 22, '(0, 255, 255)'], [6, 40, 28, '(255, 0, 0)'], [16, 6, 42, '(255, 255, 0)'], [18, 4, 36, '(255, 255, 0)'], [50, 28, 6, '(0, 0, 255)'], [36, 22, 8, '(0, 255, 255)'], [34, 16, 46, '(0, 255, 0)'], [18, 36, 10, '(255, 0, 255)'], [32, 14, 44, '(0, 255, 0)'], [36, 38, 38, '(0, 0, 0)'], [36, 8, 26, '(0, 255, 255)'], [48, 8, 14, '(0, 255, 255)'], [8, 20, 30, '(255, 255, 0)'], [12, 20, 26, '(255, 255, 255)'], [22, 38, 8, '(255, 0, 255)'], [40, 46, 50, '(0, 0, 0)'], [12, 42, 24, '(255, 0, 255)'], [40, 34, 28, '(0, 0, 0)'], [18, 46, 30, '(255, 0, 0)'], [42, 8, 22, '(0, 255, 255)'], [44, 36, 26, '(0, 0, 255)'], [30, 46, 18, '(0, 0, 255)'], [36, 50, 16, '(0, 0, 255)'], [36, 8, 28, '(0, 255, 0)'], [4, 6, 6, '(255, 255, 255)'], [50, 20, 36, '(0, 255, 0)'], [42, 44, 46, '(0, 0, 0)'], [44, 36, 28, '(0, 0, 0)'], [6, 38, 24, '(255, 0, 255)'], [28, 36, 44, '(0, 0, 0)'], [28, 8, 34, '(0, 255, 0)'], [16, 6, 42, '(255, 255, 0)'], [50, 16, 30, '(0, 255, 0)'], [36, 16, 42, '(0, 255, 0)'], [22, 26, 32, '(255, 255, 0)'], [24, 48, 26, '(255, 0, 255)'], [36, 30, 24, '(0, 0, 255)'], [36, 4, 20, '(0, 255, 255)'], [26, 26, 28, '(255, 255, 0)'], [16, 20, 24, '(255, 255, 255)'], [20, 26, 32, '(255, 255, 0)'], [28, 34, 38, '(0, 0, 0)'], [12, 32, 8, '(255, 0, 255)'], [20, 8, 40, '(255, 255, 0)'], [36, 46, 8, '(0, 0, 255)'], [22, 32, 40, '(255, 0, 0)'], [12, 24, 36, '(255, 255, 0)'], [48, 8, 14, '(0, 255, 255)'], [8, 24, 40, '(255, 255, 0)'], [48, 8, 14, '(0, 255, 255)'], [26, 18, 8, '(255, 255, 255)'], [34, 34, 34, '(0, 0, 0)'], [6, 44, 36, '(255, 0, 0)'], [10, 20, 30, '(255, 255, 0)'], [14, 28, 44, '(255, 0, 0)'], [8, 8, 8, '(255, 255, 255)'], [18, 40, 16, '(255, 0, 255)'], [30, 46, 18, '(0, 0, 255)'], [18, 48, 34, '(255, 0, 0)'], [50, 12, 20, '(0, 255, 255)'], [44, 28, 12, '(0, 0, 255)'], [40, 32, 22, '(0, 0, 255)'], [14, 48, 36, '(255, 0, 0)']]
# array = np.array(all_data)


volumetric_data=[[11520.000000000002, '255,0,255'], [7919.9999999999982, '255,255,0'], [13728.000000000011, '255,255,0'], [28000.000000000007, '0,0,255'], [10200.000000000036, '0,255,0'], [5183.9999999999891, '255,255,255'], [12799.999999999991, '255,0,255'], [55296.0, '0,255,0'], [2240.0000000000196, '255,255,255'], [2992.0000000000123, '0,255,255'], [6719.9999999999754, '255,0,0'], [4031.9999999999736, '255,255,0'], [2591.9999999999964, '255,255,0'], [8400.0000000000055, '0,0,255'], [6335.9999999999955, '0,255,255'], [25024.000000000087, '0,255,0'], [6479.9999999999982, '255,0,255'], [19712.0, '0,255,0'], [51984.000000000116, '0,0,0'], [7488.0000000001073, '0,255,255'], [5375.9999999999827, '0,255,255'], [4800.00000000001, '255,255,0'], [6240.0000000000227, '255,255,255'], [6688.0000000000209, '255,0,255'], [91999.999999999942, '0,0,0'], [12096.000000000004, '255,0,255'], [38079.999999999985, '0,0,0'], [24840.000000000036, '255,0,0'], [7391.9999999999927, '0,255,255'], [41184.000000000044, '0,0,255'], [24840.00000000004, '0,0,255'], [28799.999999999978, '0,0,255'], [8063.9999999999618, '0,255,0'], [144.00000000000031, '255,255,255'], [36000.000000000087, '0,255,0'], [85008.000000000146, '0,0,0'], [44352.000000000029, '0,0,0'], [5472.0000000000073, '255,0,255'], [44351.999999999993, '0,0,0'], [7615.9999999999882, '0,255,0'], [4032.0000000000164, '255,255,0'], [23999.999999999989, '0,255,0'], [24192.000000000055, '0,255,0'], [18304.000000000113, '255,255,0'], [29952.000000000044, '255,0,255'], [25920.00000000004, '0,0,255'], [2879.9999999999509, '0,255,255'], [18927.99999999992, '255,255,0'], [7680.0000000000018, '255,255,255'], [16639.999999999956, '255,255,0'], [36175.999999999913, '0,0,0'], [3071.9999999999955, '255,0,255'], [6399.9999999999791, '255,255,0'], [13247.999999999995, '0,0,255'], [28160.000000000124, '255,0,0'], [10368.000000000009, '255,255,0'], [5375.9999999999836, '0,255,255'], [7680.0000000000009, '255,255,0'], [5376.0000000000864, '0,255,255'], [3744.0000000000091, '255,255,255'], [39303.999999999971, '0,0,0'], [9503.9999999999909, '255,0,0'], [6000.0000000000246, '255,255,0'], [17248.000000000004, '255,0,0'], [511.99999999999807, '255,255,255'], [11519.999999999996, '255,0,255'], [24839.999999999993, '0,0,255'], [29375.999999999964, '255,0,0'], [12000.000000000013, '0,255,255'], [14783.999999999975, '0,0,255'], [28160.000000000004, '0,0,255'], [24191.999999999953, '255,0,0']]

colors_8=[]
volume=[]
for i in range(len(volumetric_data)):
    volume.append(volumetric_data[i][0])
    if volumetric_data[i][1]=="0,0,0":
        colors_8.append(1)
    elif volumetric_data[i][1]=="255,0,0":
        colors_8.append(2)    
    elif volumetric_data[i][1]=="0,255,0":
        colors_8.append(3)        
    elif volumetric_data[i][1]=="0,0,255":
        colors_8.append(4)            
    elif volumetric_data[i][1]=="255,255,0":
        colors_8.append(5)            
    elif volumetric_data[i][1]=="0,255,255":
        colors_8.append(6)
    elif volumetric_data[i][1]=="255,0,255":
        colors_8.append(7)
    elif volumetric_data[i][1]=="255,255,255":
        colors_8.append(8)
        
x_lengths=list(list(zip(*all_data))[0])
y_lengths=list(list(zip(*all_data))[1])
z_lengths=list(list(zip(*all_data))[2])
r_val=list(list(zip(*all_data))[3])
g_val=list(list(zip(*all_data))[4])
b_val=list(list(zip(*all_data))[5])


fig, ax = plt.subplots()
ax.scatter(x=volume,y=colors_8)
ax.set_title('Volume/8 colors')
plt.show()


fig, ax = plt.subplots()
ax.scatter(x=x_lengths,y=r_val)
ax.set_title('x/r')
plt.show()

fig, ax = plt.subplots()
ax.scatter(x=y_lengths,y=g_val)
ax.set_title('y/g')
plt.show()

fig, ax = plt.subplots()
ax.scatter(x=z_lengths,y=b_val)
ax.set_title('z/b')
plt.show()


"""X-R"""
X_train, X_validation, Y_train, Y_validation = train_test_split(x_lengths, r_val, test_size=0.25, random_state=8)
mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

myline = np.linspace(min(x_lengths), max(x_lengths),100)
plt.scatter(X_train, Y_train)
plt.plot(myline, mymodel(myline))
plt.title("x-r")
plt.show()


from sklearn.metrics import r2_score

r2 = r2_score(Y_train, mymodel(X_train))
r2 = r2_score(Y_validation, mymodel(X_validation))
# print(Y_validation)
result_m=[]
TP=0
TN=0
FP=0
FN=0
for i in range(len(X_validation)):
    if mymodel(X_validation[i])<(255/2):
        result=0
        if result==Y_validation[i]:
            TN+=1
        else:
            FN+=1
    else:
        result=255
        if result==Y_validation[i]:
            TP+=1
        else:
            FP+=1
    
    result_m.append([result,Y_validation[i]])
print(TP,TN,FP,FN)
# print( result_m)


"""Y-G"""
X_train, X_validation, Y_train, Y_validation = train_test_split(y_lengths, g_val, test_size=0.25, random_state=8)
mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

myline = np.linspace(min(y_lengths), max(y_lengths),100)
plt.scatter(X_train, Y_train)
plt.plot(myline, mymodel(myline))
plt.title("y-g")
plt.show()


r2 = r2_score(Y_train, mymodel(X_train))
r2 = r2_score(Y_validation, mymodel(X_validation))
# print(Y_validation)
result_m2=[]
TP=0
TN=0
FP=0
FN=0
for i in range(len(X_validation)):
    if mymodel(X_validation[i])<(255/2):
        result=0
        if result==Y_validation[i]:
            TN+=1
        else:
            FN+=1
    else:
        result=255
        if result==Y_validation[i]:
            TP+=1
        else:
            FP+=1
    
    result_m2.append([result,Y_validation[i]])
print(TP,TN,FP,FN)
# print( result_m2)


"""Z-B"""
X_train, X_validation, Y_train, Y_validation = train_test_split(z_lengths, b_val, test_size=0.25, random_state=8)
mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

myline = np.linspace(min(z_lengths), max(z_lengths),100)
plt.scatter(X_train, Y_train)
plt.plot(myline, mymodel(myline))
plt.title("z-b")
plt.show()


r2 = r2_score(Y_train, mymodel(X_train))
r2 = r2_score(Y_validation, mymodel(X_validation))
# print(Y_validation)
result_m3=[]
TP=0
TN=0
FP=0
FN=0
for i in range(len(X_validation)):
    if mymodel(X_validation[i])<(255/2):
        result=0
        if result==Y_validation[i]:
            TN+=1
        else:
            FN+=1
    else:
        result=255
        if result==Y_validation[i]:
            TP+=1
        else:
            FP+=1
    
    result_m3.append([result,Y_validation[i]])
print(TP,TN,FP,FN)
# print( result_m3)
r=list(zip(*result_m))[0]
g=list(zip(*result_m2))[0]
b=list(zip(*result_m3))[0]
print(list(zip(r,g,b)))
        
#https://www.w3schools.com/python/python_ml_train_test.asp

"""CHALLENGE 2"""
# all_data=[[20, 44, 22, 89, 222, 100], [18, 30, 44, 78, 144, 222], [16, 48, 34, 67, 244, 166], [32, 42, 6, 155, 211, 11], [26, 46, 20, 122, 233, 89], [46, 40, 36, 233, 200, 177], [10, 32, 6, 33, 155, 11], [48, 24, 48, 244, 111, 244], [32, 38, 42, 155, 188, 211], [10, 26, 44, 33, 122, 222], [48, 36, 24, 244, 177, 111], [22, 10, 46, 100, 33, 233], [8, 38, 24, 22, 188, 111], [8, 32, 10, 22, 155, 33], [26, 12, 44, 122, 44, 222], [12, 40, 22, 44, 200, 100], [48, 22, 42, 244, 100, 211], [8, 36, 20, 22, 177, 89], [42, 42, 44, 211, 211, 222], [16, 34, 6, 67, 166, 11], [26, 32, 38, 122, 155, 188], [8, 18, 30, 22, 78, 144], [42, 4, 12, 211, 0, 44], [40, 10, 26, 200, 33, 122], [12, 18, 22, 44, 78, 100], [38, 22, 6, 188, 100, 11], [14, 8, 48, 55, 22, 244], [40, 22, 6, 200, 100, 11], [46, 14, 26, 233, 55, 122], [8, 46, 38, 22, 233, 188], [6, 22, 40, 11, 100, 200], [40, 6, 18, 200, 11, 78], [18, 36, 10, 78, 177, 33], [10, 10, 12, 33, 33, 44], [22, 38, 8, 100, 188, 22], [50, 6, 8, 255, 11, 22], [12, 4, 42, 44, 0, 211], [40, 26, 10, 200, 122, 33], [14, 22, 28, 55, 100, 133], [40, 20, 46, 200, 89, 233], [20, 10, 44, 89, 33, 222], [8, 22, 34, 22, 100, 166], [50, 30, 10, 255, 144, 33], [42, 48, 8, 211, 244, 22], [28, 8, 32, 133, 22, 155], [14, 8, 48, 55, 22, 244], [6, 22, 38, 11, 100, 188], [28, 30, 30, 133, 144, 144], [34, 38, 42, 166, 188, 211], [44, 4, 10, 222, 0, 33], [50, 8, 14, 255, 22, 55], [48, 24, 44, 244, 111, 222], [16, 50, 36, 67, 255, 177], [28, 36, 46, 133, 177, 233], [14, 22, 32, 55, 100, 155], [16, 28, 42, 67, 133, 211], [24, 30, 36, 111, 144, 177], [26, 42, 12, 122, 211, 44], [24, 30, 34, 111, 144, 166], [38, 28, 20, 188, 133, 89], [36, 36, 38, 177, 177, 188], [20, 12, 4, 89, 44, 0], [6, 16, 26, 11, 67, 122], [26, 42, 10, 122, 211, 33], [8, 8, 8, 22, 22, 22], [14, 36, 10, 55, 177, 33], [6, 24, 42, 11, 111, 211], [48, 32, 16, 244, 155, 67], [36, 44, 6, 177, 222, 11], [28, 12, 42, 133, 44, 211], [50, 40, 30, 255, 200, 144], [12, 46, 34, 44, 233, 166]]
# array = np.array(all_data)
      
# x_lengths=list(list(zip(*all_data))[0])
# y_lengths=list(list(zip(*all_data))[1])
# z_lengths=list(list(zip(*all_data))[2])
# r_val=list(list(zip(*all_data))[3])
# g_val=list(list(zip(*all_data))[4])
# b_val=list(list(zip(*all_data))[5])

# fig, ax = plt.subplots()
# ax.scatter(x=x_lengths,y=r_val)
# ax.set_title('x/r')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x=y_lengths,y=g_val)
# ax.set_title('y/g')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x=z_lengths,y=b_val)
# ax.set_title('z/b')
# plt.show()


# """X-R"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(x_lengths, r_val, test_size=0.16, random_state=3)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(x_lengths), max(x_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("x-r")
# plt.show()


# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))
# # print(r2)

# result_m=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.8*Y_validation[i]) and result<(1.2*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m.append([result,Y_validation[i],pr])

# # print( result_m)


# """Y-G"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(y_lengths, g_val, test_size=0.16, random_state=3)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(y_lengths), max(y_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("y-g")
# plt.show()


# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))
# # # print(Y_validation)
# result_m2=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.8*Y_validation[i]) and result<(1.2*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m2.append([result,Y_validation[i],pr])

# # print( result_m2)

# """Z-B"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(z_lengths, b_val, test_size=0.16, random_state=3)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(z_lengths), max(z_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("z-b")
# plt.show()

# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))

# result_m3=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.8*Y_validation[i]) and result<(1.2*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m3.append([result,Y_validation[i],pr])

# # print( result_m3)


# r=list(zip(*result_m))[0]
# g=list(zip(*result_m2))[0]
# b=list(zip(*result_m3))[0]
# print("Predicted")
# for x in list(zip(r,g,b)):
#     print(x)
# # print(list(zip(r,g,b)))

# r_a=list(zip(*result_m))[1]
# g_a=list(zip(*result_m2))[1]
# b_a=list(zip(*result_m3))[1]
# print("Actual")
# for x in list(zip(r_a,g_a,b_a)):
#     print(x)


# r_c=list(zip(*result_m))[2]
# g_c=list(zip(*result_m2))[2]
# b_c=list(zip(*result_m3))[2]
# print("Correctness")
# for x in list(zip(r_c,g_c,b_c)):
#     print(x)


# """CHALLENGE 3"""
# all_data=[[20, 44, 22, 89, 33, 100], [18, 30, 44, 78, 111, 222], [16, 48, 34, 67, 11, 166], [32, 42, 6, 155, 44, 11], [26, 46, 20, 122, 22, 89], [46, 40, 36, 233, 55, 177], [10, 32, 6, 33, 100, 11], [48, 24, 48, 244, 144, 244], [32, 38, 42, 155, 67, 211], [10, 26, 44, 33, 133, 222], [48, 36, 24, 244, 78, 111], [22, 10, 46, 100, 222, 233], [8, 38, 24, 22, 67, 111], [8, 32, 10, 22, 100, 33], [26, 12, 44, 122, 211, 222], [12, 40, 22, 44, 55, 100], [48, 22, 42, 244, 155, 211], [8, 36, 20, 22, 78, 89], [42, 42, 44, 211, 44, 222], [16, 34, 6, 67, 89, 11], [26, 32, 38, 122, 100, 188], [8, 18, 30, 22, 177, 144], [42, 4, 12, 211, 255, 44], [40, 10, 26, 200, 222, 122], [12, 18, 22, 44, 177, 100], [38, 22, 6, 188, 155, 11], [14, 8, 48, 55, 233, 244], [40, 22, 6, 200, 155, 11], [46, 14, 26, 233, 200, 122], [8, 46, 38, 22, 22, 188], [6, 22, 40, 11, 155, 200], [40, 6, 18, 200, 244, 78], [18, 36, 10, 78, 78, 33], [10, 10, 12, 33, 222, 44], [22, 38, 8, 100, 67, 22], [50, 6, 8, 255, 244, 22], [12, 4, 42, 44, 255, 211], [40, 26, 10, 200, 133, 33], [14, 22, 28, 55, 155, 133], [40, 20, 46, 200, 166, 233], [20, 10, 44, 89, 222, 222], [8, 22, 34, 22, 155, 166], [50, 30, 10, 255, 111, 33], [42, 48, 8, 211, 11, 22], [28, 8, 32, 133, 233, 155], [14, 8, 48, 55, 233, 244], [6, 22, 38, 11, 155, 188], [28, 30, 30, 133, 111, 144], [34, 38, 42, 166, 67, 211], [44, 4, 10, 222, 255, 33], [50, 8, 14, 255, 233, 55], [48, 24, 44, 244, 144, 222], [16, 50, 36, 67, 0, 177], [28, 36, 46, 133, 78, 233], [14, 22, 32, 55, 155, 155], [16, 28, 42, 67, 122, 211], [24, 30, 36, 111, 111, 177], [26, 42, 12, 122, 44, 44], [24, 30, 34, 111, 111, 166], [38, 28, 20, 188, 122, 89], [36, 36, 38, 177, 78, 188], [20, 12, 4, 89, 211, 0], [6, 16, 26, 11, 188, 122], [26, 42, 10, 122, 44, 33], [8, 8, 8, 22, 233, 22], [14, 36, 10, 55, 78, 33], [6, 24, 42, 11, 144, 211], [48, 32, 16, 244, 100, 67], [36, 44, 6, 177, 33, 11], [28, 12, 42, 133, 211, 211], [50, 40, 30, 255, 55, 144], [12, 46, 34, 44, 22, 166]]


      
# x_lengths=list(list(zip(*all_data))[0])
# y_lengths=list(list(zip(*all_data))[1])
# z_lengths=list(list(zip(*all_data))[2])
# r_val=list(list(zip(*all_data))[3])
# g_val=list(list(zip(*all_data))[4])
# b_val=list(list(zip(*all_data))[5])


# fig, ax = plt.subplots()
# ax.scatter(x=x_lengths,y=r_val)
# ax.set_title('x/r')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x=y_lengths,y=g_val)
# ax.set_title('y/g')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(x=z_lengths,y=b_val)
# ax.set_title('z/b')
# plt.show()


# """X-R"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(x_lengths, r_val, test_size=0.12, random_state=7)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(x_lengths), max(x_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("x-r")
# plt.show()


# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))
# # print(r2)

# result_m=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.9*Y_validation[i]) and result<(1.1*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m.append([result,Y_validation[i],pr])

# # print( result_m)


# """Y-G"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(y_lengths, g_val, test_size=0.12, random_state=7)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(y_lengths), max(y_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("y-g")
# plt.show()


# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))
# print(len(Y_validation))
# result_m2=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.9*Y_validation[i]) and result<(1.1*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m2.append([result,Y_validation[i],pr])

# # print( result_m2)

# """Z-B"""
# X_train, X_validation, Y_train, Y_validation = train_test_split(z_lengths, b_val, test_size=0.12, random_state=7)
# mymodel = np.poly1d(np.polyfit(X_train, Y_train, 4))

# myline = np.linspace(min(z_lengths), max(z_lengths),100)
# plt.scatter(X_train, Y_train)
# plt.plot(myline, mymodel(myline))
# plt.title("z-b")
# plt.show()

# r2 = r2_score(Y_train, mymodel(X_train))
# r2 = r2_score(Y_validation, mymodel(X_validation))

# result_m3=[]
# for i in range(len(X_validation)):
#     result=round(mymodel(X_validation[i]),2)
#     if result>(0.9*Y_validation[i]) and result<(1.1*Y_validation[i]):
#         pr=True
#     else:
#         pr=False
#     result_m3.append([result,Y_validation[i],pr])

# # print( result_m3)

# r=list(zip(*result_m))[0]
# g=list(zip(*result_m2))[0]
# b=list(zip(*result_m3))[0]
# print("Predicted")
# for x in list(zip(r,g,b)):
#     print(x)
# # print(list(zip(r,g,b)))

# r_a=list(zip(*result_m))[1]
# g_a=list(zip(*result_m2))[1]
# b_a=list(zip(*result_m3))[1]
# print("Actual")
# for x in list(zip(r_a,g_a,b_a)):
#     print(x)


# r_c=list(zip(*result_m))[2]
# g_c=list(zip(*result_m2))[2]
# b_c=list(zip(*result_m3))[2]
# print("Correctness")
# for x in list(zip(r_c,g_c,b_c)):
#     print(x)




# """MULTIPLE REGRESSION"""

array = np.array(all_data)
x = array[:,0:3]
y = array[:,3:4]
#r value
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.12, random_state=7)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_validation)

from sklearn.metrics import r2_score
score=r2_score(Y_validation,Y_pred)

Y_validation_list = Y_validation.tolist()
Y_pred_list = Y_pred.tolist()
print(list(zip(Y_validation_list,Y_pred_list)))
print(score)




