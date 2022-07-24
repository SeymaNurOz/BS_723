# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:35:08 2022

@author: seyma
15 sec intervals
4*5=20 intervals

1:-16.45
        red 1
        white 4
        grey 1
        bordo 1
        
 
2:-16.30
        bordo 2
        white 3
        grey 3
        
        other side
        1 balck other road

3:-16.15
        1 white turning 
        4 grey 
        1 red 
        2 black 
        
        other side
        1 black
        1 grey

4:-16.00
        1 white turning 
        4 grey
        2 black
        1.5 bicycle
        
5:-15.45
        1 white turning 
        2 grey
        1 black
        1 white   
        
        other side
        3 pedesterians
        
6:-15.30
        1 white turning 
        4 black
        
        other side
        3 pedesterians
        
7:-15.15
        1 white turning 
        1 grey-green turning 
        1 blue
        1 white
        2 motors
        
        other
        1 red
   
8:-15.00
        1 red
        1 white
        3 motors
        1 grey
        5 pedesterian 
        1 pet
        
9:-14.45
        2 black
        2 grey
        5 pedesterian
        2 pet
        
        other side
        3 motors
        1 white
        
10:-14.30
        2 grey
        2 black
        5 pedesterian
        
        other side
        1 black
        
11:-14.15
        3 black
        1 bordo
        5 pedesterian
        1 pet
12:-14.00
        1 red 
        1 black
        1 grey
        2 pedesterian
        1 pet
        
        other side
        1 brown
        1 white
        
13:-13.45
        1 black
        1 grey
        3 pedesterian
        1 pet  
        
14:-13.30
        1 red turning 
        2 white 
        1 grey
        2 bicycle
        1 pedesterian
        1 white entering
        
15:-13.15
        1 white 
        2 black
        1 motor
        1 pedesterian
        1 white entering
        
        other side
        1 black
        
16:-13.00
        1 white 
        1 black
        2 grey
        2 pedesterian
        1 white entering
        
17:-12.45
        2 white 
        2 grey
        1 brown
        3 pedesterian

18:-12.30
        2 grey
        1 brown
        1 black
        4 pedesterian   
        1 pet
        
19:-12.15
        3 grey 
        1 white
        1 black
        1 pedesterian
        1 pet
        
        other side
        1 red
        
20:-12.00
        1 grey 
        2 black
        2 white
        1 pedesterian
        1 pet
        
        other side
        1 black
        
        
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
names=[]
list_type=["white_car","black_car","grey_car",
                "red_car","bordo_car","other_car",
                "motor","bicycle,","pedesterian","pet"]
names+=list_type

# list_type=[0="white_car",1="black_car",2=grey_car,
#                3=red_car,4=bordo_car,5=other(blue-brown_car,
#                6=motor,7=bicycle,8=pedesterian,9=pet]

def avarage(list_x):
    sum=0
    for i in list_x:
        sum+=i
    if len(list_x)>0:
        return sum/len(list_x)
    
def existance_per(list_x):
    pos=0
    for i in list_x:
        if i>0:
            pos+=1
    if len(list_x)>0:
        return pos*100/len(list_x)
        
all_dict={1:[4,0,1,1,1,0,0,0,0,0],
          2:[3,1,3,0,2,0,0,0,0,0],
          3:[1,3,5,1,0,0,0,0,0,0],
          4:[1,2,6,0,0,0,0,1,0,0],
          5:[2,1,2,0,0,0,0,0,3,0],
          6:[1,4,0,0,0,0,0,0,3,0],
          7:[2,0,1,1,0,1,1,0,0,0],
          8:[1,0,1,1,0,0,3,0,5,1],
          9:[1,2,2,0,0,0,3,0,5,2],
          10:[0,3,2,0,0,0,0,0,5,0],
          
          11:[0,3,0,0,1,0,0,0,5,1],
          12:[1,1,1,1,0,1,0,0,2,1],
          13:[0,1,1,0,0,0,0,0,3,1],
          14:[3,0,1,1,0,0,0,2,1,0],
          15:[2,3,0,0,0,0,1,0,1,0],
          16:[2,1,2,0,0,0,0,0,2,0],
          17:[2,0,2,0,0,1,0,0,3,0],
          18:[0,1,2,0,0,1,0,0,4,1],
          19:[1,1,3,1,0,0,0,0,1,1],
          20:[2,3,1,0,0,0,0,0,1,1]
          }
# fig, axes = plt.subplots(1,10)
avarage_l=[]
exist_l=[]
for i in range(len(list_type)):
    list_type[i]=[]
    
    k=0
    while k<20:
        k+=1
        list_type[i].append(all_dict[k][i])
    # print(list_type[i])
    avarage_l.append(avarage(list_type[i]))
    exist_l.append(existance_per(list_type[i])) 
    # plt.scatter(range(0,20), list_type[i])
    # plt.plot(range(0,20), list_type[i],label=names[i])
# print(avarage_l)


z=zip(exist_l,names)
sorted_zip=sorted(z)
print(sorted_zip)
for i in range(len(exist_l)):
    print("%{} of the frames includes {}.".format(sorted_zip[i][0],sorted_zip[i][1]))

plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(names, exist_l, align='center',color="maroon")

ax.set_ylabel('Types')
ax.set_xlabel('Percentage')
ax.set_title('How many of the interval includes the type(%)')



plt.rcdefaults()
fig, ax = plt.subplots()
ax.barh(names, avarage_l, align='center')

ax.set_ylabel('Types')
ax.set_xlabel('Avarage num')
ax.set_title('Avarage number of each type in all intervals')

ax.legend()


plt.show()







# xs = df['x'][:].values
# ys = df['y'][:].values
# # Plot y vs. x
# plt.scatter(xs, ys)
# # Set the labels for x and y axes:
# plt.xlabel('x')
# plt.ylabel('y')
# # Set the title of the figure
# plt.title("y vs x")
# Text(0.5, 1.0, 'y vs x')

# # Create a 2x2 grid of plots
# fig, axes = plt.subplots(5,2)
# # Plot (1,1)
# axes[0,0].plot(x, x)
# axes[0,0].set_title("$y=x$")
# # Plot (1,2)
# axes[0,1].plot(x, x**2)
# axes[0,1].set_title("$y=x^2$")
# # Plot (2,1)
# axes[1,0].plot(x, x**3)
# axes[1,0].set_title("$y=x^3$")
# # Plot (2,2)
# axes[1,1].plot(x, x**4)
# axes[1,1].set_title("$y=x^4$")
# # Adjust vertical space between rows
# plt.subplots_adjust(hspace=1)


# # Uniformly sample 50 x values between -2 and 2:
# x = np.linspace(0, 6, 50)
# # Plot y = x
# plt.plot(x, x, label='$y=x$')
# # Plot y = x^2
# plt.plot(x, x**2, label='$y=x^2$')
# # Plot y = x^3
# plt.plot(x, x**3, label='$y=x^3$')
# # Set the labels for x and y axes:
# plt.xlabel('x')
# plt.ylabel('y')
# # Set the title of the figure
# plt.title("Our First Plot -- Pyplot Style")
# # Create a legend
# plt.legend()







