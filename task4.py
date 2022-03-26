
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths

from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[21]:


import tensorflow as tf


# In[2]:


dataset=r'D://BISM/task4/Mask Dataset'


# In[3]:


imagepath=list(paths.list_images(dataset))


# In[4]:


data=[]
labels=[]

for i in imagepath:
    
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)
    


# In[5]:


data=np.array(data,dtype='float32')
labels=np.array(labels)


# In[7]:


lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[11]:


labels


# In[12]:


(X_train,X_test,Y_train, Y_test)=train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=10,stratify=labels)


# In[13]:


aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')


# In[15]:


baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[16]:


print(baseModel.summary())


# In[27]:


headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2, activation = (tf.nn.softmax)) (headModel)
#headModel=Dense(2,activation='softmax')(headModel)
#model.add(Dense(81, activation = (tf.nn.softmax)))


# In[28]:


model=Model(inputs=baseModel.input,outputs=headModel)


# In[29]:


for layer in baseModel.layers:
    layer.trainable=False


print(model.summary())


# In[56]:


aug.flow(X_train,Y_train,batch_size=BS)


# In[58]:


learning_rate=0.001
Epochs=1
BS=10
opt=Adam(lr=learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

augmanted_data=model.fit_generator(
    aug.flow(X_train,Y_train,batch_size=BS),
    steps_per_epoch=len(X_train)//BS,
    validation_data=(X_test,Y_test),
    validation_steps=len(X_test)//BS,
    epochs=Epochs
)


# In[59]:


model.save(r'D://BISM/task4/mobilenet_v2.model')


# In[61]:


predict=model.predict(X_test,batch_size=BS)
predict=np.argmax(predict,axis=1)
#print(classification_report(Y_test.argmax(axis=1),predict,target_names=lb.classes_))


# In[63]:


print(classification_report(Y_test.argmax(axis=1),predict))


# In[67]:


N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), augmanted_data.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), augmanted_data.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), augmanted_data.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), augmanted_data.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(r'D://BISM/task4/Mask Dataset/plot_v2.png')


# In[69]:


from keras.models import load_model
import cv2


# In[103]:


weightpaths=os.path.sep.join([r'D:\BISM\task4','res10_300x300_ssd_iter_140000.caffemodel'])


# In[110]:


protoypes=os.path.sep.join([r'D:\BISM\task4','deploy.prototxt'])


# In[111]:


weightpaths


# In[112]:


net=cv2.dnn.readNet(protoypes,weightpaths)


# In[114]:


model.summary()


# In[127]:


image=cv2.imread(r'D:\BISM\task4\download.jpg')


# In[128]:


type(image)


# In[129]:


plt.imshow(image)


# In[130]:


(h,w)=image.shape[:2]


# In[131]:


(h,w)


# In[132]:


blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))


# In[133]:


blob.shape


# In[134]:


net.setInput(blob)
detections=net.forward()


# In[135]:


detections.shape


# In[136]:


#loop over the detections
for i in range(0,detections.shape[2]):
    confidence=detections[0,0,i,2]
    
    
    if confidence>0.5:
        #we need the X,Y coordinates
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype('int')
        
        #ensure the bounding boxes fall within the dimensions of the frame
        (startX,startY)=(max(0,startX),max(0,startY))
        (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
        
        #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
        face=image[startY:endY, startX:endX]
        face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        face=cv2.resize(face,(224,224))
        face=img_to_array(face)
        face=preprocess_input(face)
        face=np.expand_dims(face,axis=0)
        
        (mask,withoutMask)=model.predict(face)[0]
        
        #determine the class label and color we will use to draw the bounding box and text
        label='Mask' if mask>withoutMask else 'No Mask'
        color=(0,255,0) if label=='Mask' else (0,0,255)
        
        #include the probability in the label
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)
        
        #display the label and bounding boxes
        cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        cv2.rectangle(image,(startX,startY),(endX,endY),color,2)
        
        
        
cv2.imshow("OutPut",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

