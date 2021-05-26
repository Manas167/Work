
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

class myfirstclass:
    pass


# In[5]:

a = myfirstclass()
b = myfirstclass()


# In[29]:

class caldistance1:
    def move(self,x,y):
        self.x = x
        self.y = y
    def distance(self,anotherpoint):
        return abs(self.x - anotherpoint.x)


# In[40]:

a = caldistance1()
b = caldistance1()
a.move(2,3)
b.move(3,4)
assert a.distance(b) == b.distance(a)
print("Arrived")


# In[31]:

class caldistance2:
    def __init__(self,x,y):
        self.move(x,y)
    def move(self,x,y):
        self.x = x
        self.y = y
    def distance(self,anotherpoint):
        return abs(self.x - anotherpoint.x)


# In[33]:

a = caldistance2(0,0)
b = caldistance2(1,1)


# In[48]:

class caldistance3:
    
    def __init__(self,x=0,y=0):
        self.move(x,y)
    def move(self,x,y):
        'moves point to a different location'
        self.x = x
        self.y = y
    def distance(self,anotherpoint):
        'calculate x distance between two points'
        return abs(self.x - anotherpoint.x)


# In[49]:

a = caldistance3()
b = caldistance3(1,1)


# In[50]:

help(caldistance3)


# In[ ]:



