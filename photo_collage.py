import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

image_names=os.listdir("images")

pokemons=[]

for f in image_names:
    if f.endswith('.jpg') or f.endswith('.jpeg'):
        file_path="images/"+f
        img=cv2.imread(file_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(200,200))
        pokemons.append(img)

img=pokemons[2]
img=cv2.resize(img,(100,100))

border=np.zeros((10,410,3),dtype='int')
border[:,:,0]=0
border2=np.zeros((430,10,3),dtype='int')
border2[:,:,0]=0
border3=np.zeros((200,10,3),dtype='int')
border3[:,:,0]=0
top=np.hstack((pokemons[3],border3,pokemons[4]))
plt.imshow(top)
plt.show()
bottom=np.hstack((pokemons[0],border3,pokemons[1]))
collage=np.vstack((border,top,border,bottom,border))
img=cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT)
collage=np.hstack((border2,collage,border2))
print(collage.shape)
collage[155:275,155:275]=img
plt.imshow(collage)
plt.show()

collage=np.reshape(collage,(430*430,3))
print(collage.shape)
print(img.shape)

import pandas as pd
df=pd.DataFrame(collage)
df.to_csv('Collage.csv',index=False,header=('r','g','b'))