#!/usr/bin/env python
# coding: utf-8

# ## **Data Science in Agriculture. Land use classification**
# 

# ## Abstract
# 

# In this laboratory, we will explore how land use types can be classified based on aerial and satellite photographs of the earth's surface. We will learn to build our own photo masks based on ready-made masks and display them on the screen.
# 

# ## Introduction
# 

# Determining how land is used is a huge problem today. After all, its improper and illegal use can lead to both economic and natural disasters. One of the ways to assess the use is the analysis of aerial and satellite images of the earth's surface. A big problem is to build a mathematical model that can determine the type of land use based on colors. If you have ready-made photos and masks of land use, you can use the methods of artificial intelligence and big data to build a model-classifier.
# 
# In this lab, we will learn how to upload photos, transform pixels and colors into a data set. Then we will learn how to build a classifier and predict land use masks based on it.
# 

# ## Import the required libraries
# 

# We will use libraries os and glob to work with files and folders.
# We will also use Matplotlib and Seaborn for visualizing our dataset to gain a better understanding of the images we are going to be handling.
# NumPy will be used for arrays of images. Scikit-Learn - for classical classification models. Pandas - for DataSet creation.
# 

# In[1]:


import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import os
from glob import glob
import json
from PIL import Image
from colormap import rgb2hex, hex2rgb

#Classifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import plot_confusion_matrix


# ## Loading of data
# 

# For convenience, let's create a function that downloads all and displays the last images of land and masks from a specified directory.
# All training pictures and their masks have to be placed in differend directories.
# Separate a csv file that contains the description of land use classes.
# 
# The function has to work in the following way:
# 
# 1.  Download a csv file **[json.load()](https://docs.python.org/3/library/json.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturelanduseclassification28296414-2022-01-01)** and display classes description.
# 2.  Download all images from a specific directory:
# 
# **[Image.open()](https://pillow.readthedocs.io/en/stable/reference/Image.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturelanduseclassification28296414-2022-01-01)**.
# 3\. Display the last image + mask.
# 4\. Form and return a DataSet that has to be an array of tuples \[image, mask].
# 
# Remark: the downloaded directories contain 72 images and their masks located in different subfolders according to image resolutions. Due to the fact that the dataset for training will represent each point in the form of a separate record - the size of the dataset will be too large to conduct the training on a local computer. Therefore, we will select a separate folder that contains 9 photos and their masks.
# 

# In[2]:


cd "F:\Artifitial Intelligence\projects\data analysis\Agricaluture\archive"


# In[3]:


def get_data(folder, file):
    # download json
    f = open(folder + "/" + file,)
    data = json.load(f)
    f.close()
    cl = {}
    # Create a dictionary with classes
    for i, c in enumerate(data['classes']):
        cl[i] = dict(c)
        
    for k, v in cl.items():
        print('Class', k)
        for k2, v2 in v.items():
            print("   ", k2, v2)
    data = []
    
    # download images
    sd = [item for item in os.listdir(folder) if os.path.isdir(folder + '/' + item)] # a list of subdirectories
    print("Subdirectories: ", sd)
    for f in sd[2:3]: #choose one of the subdirectories to download
        print("Downloading: ", f)
        images = glob(folder + "/"  +f+ "/images" + "/*.jpg") # create a list of image files

        for im in images:
            mask_f = im.replace("images", "masks").replace("jpg", "png") # create a list of mask files
            image = Image.open(im) 
            mask = Image.open(mask_f)
            if len(np.array(mask).shape) > 2:
                data.append([image, mask])
        fig = plt.figure(figsize = (10,10)) #display the last image + mask
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.show()

    return (data)


# In[4]:


d = "Semantic segmentation dataset"
f = "classes.json"
data = get_data(d, f)


# As you can see, the csv file contains a description of 6 land use classes. Each class has its own color in the mask file.
# We downloaded all the pictures and their masks from a separate directory and formed a set of data lists consisting of tuples of picture-mask.
# 

# # DataSet creation
# 

# Let's see how many images we downloaded:
# 

# In[5]:


len(data)


# As you can see, we dowloaded 9 images.
# Let's use first 4 images and masks for training and last 5 - for tests.
# 
# Let's make a function that will create a DataSet.
# 
# Each image is a set of points. Each point is represented by a tuple of RGB (red, green, blue). Every color is a number \[0-1) for float or \[0, 255) for int.
# Therefore, every image is a 3D array (height, width, color). Or a 2D array for gray scale.
# 
# To establish the dependence of color -> land use class, we need to convert each image into a dataset of the form (r, g, b) -> class.
# 
# To do this, we need to transform the image into a color matrix **[np.asarray()](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturelanduseclassification28296414-2022-01-01)**, and then transform it into a one-dimensional form **[np.flatten()](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturelanduseclassification28296414-2022-01-01)**.
# To construct the output field, we need to additionally convert the color tuple (r, g, b) from the mask file into hex format: **[rgb2hex()](https://pythonhosted.org/colormap/references.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsdatascienceinagriculturelanduseclassification28296414-2022-01-01)**.
# 

# In[6]:


def create_DataSet(data):
    DS = pd.DataFrame()
    for image, mask in data:
        # transform image to matrix
        im = np.asarray(image) 
        mk = np.asarray(mask)
        # transform a one-dimension array of r, g, b colors
        red = im[:,:,0].flatten()
        green = im[:,:,1].flatten()
        blue = im[:,:,2].flatten()
        im_f = np.array([red, green, blue])
        red = mk[:,:,0].flatten()
        green =mk[:,:,1].flatten()
        blue = mk[:,:,2].flatten()
        # calculate hex classes
        h = np.array([rgb2hex(*m) for m in zip(red, green, blue)])
        mk_f = np.array([red, green, blue, h])      
        d = np.concatenate((im_f, mk_f), axis=0)
        # create a DataSet
        DS_new = pd.DataFrame(np.transpose(d), columns = ['Im_Red', 'Im_Green', 'Im_Blue', 'Mk_Red', 'Mk_Green', 'Mk_Blue', 'HEX'])
        if len(DS) == 0:
            DS = DS_new
        else:
            DS = DS.append(DS_new)
    return DS


# <details><summary>Click <b>here</b> for the solution</summary> <code>
#         red = mk[:,:,0].flatten()
#         green = mk[:,:,1].flatten()
#         blue = mk[:,:,2].flatten()
# </code></details>
# 

# In[7]:


print("Create a training DataSet")
train = create_DataSet(data[:4])
print(train)
print("Create a test DataSet")
test = create_DataSet(data[4:])
print(test)


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# print("Create test DataSet")
# test = create_DataSet(data[4:])
# print(test)
# </code></details>
# 

# As you can see, our training DataSet consists of 2 050 681 rows and 7 columns. The test one consists of 2 566 340 rows.
# 
# Let's study the column types of training and test DataSets.
# 

# In[8]:


train.info()


# In[9]:


test.info()


# As you can see, all the columns have object type.
# 
# The last column 'HEX' contains colors in hex format. Therefore, it is necessary to change the type of this data to categorical.
# 

# In[10]:


train.loc[:, 'HEX'] = train['HEX'].astype('category')
train['HEX']


# In[11]:


test.loc[:, 'HEX'] =  test['HEX'].astype('category')
test['HEX']


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# test.loc[:, 'HEX'] = test['HEX'].astype('category')
# test['HEX']
# </code></details>
# 

# All other columns contain colors in int format, therefore we should change their types:
# 

# In[12]:


cl =['Im_Red', 'Im_Green', 'Im_Blue', 'Mk_Red', 'Mk_Green', 'Mk_Blue']
train[cl] = train[cl].astype('int64')
test[cl] = test[cl].astype('int64')
print (train.info())
print (test.info())


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# cl = ['Im_Red', 'Im_Green', 'Im_Blue', 'Mk_Red', 'Mk_Green', 'Mk_Blue']
# train[cl] = train[cl].astype('int64')
# test[cl] = test[cl].astype('int64')
# print (train.info())
# print (test.info())
# </code></details>
# 

# Let’s visualize our data and see what exactly we are working with. We use seaborn to plot the number of pixel classes and you can see what the output looks like.
# 

# In[13]:


c = pd.DataFrame(train['HEX'])
sns.set_style('darkgrid')
sns.countplot(x="HEX", data=c)


# In[14]:


c = pd.DataFrame( test['HEX'])
sns.set_style('darkgrid')
sns.countplot( x="HEX", data=c)


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# c = pd.DataFrame(test['HEX'])
# sns.set_style('darkgrid')
# sns.countplot(x="HEX", data=c)
# </code></details>
# 

# As you can see, the DataSet consist of 6 land use classes.
# 

# The training and test DataSets contain similar distribution of images classes.
# 

# # Classification model creation
# 

# In[16]:


clf = LogisticRegression(max_iter=100, n_jobs=-1)
c = train.columns
clf.fit(train[c[0:3]], train[c[-1:]].values.ravel())


# <details><summary>Click <b>here</b> for the solution</summary> <code>
# clf = LogisticRegression(max_iter=100, n_jobs=-1)
# c = train.columns
# clf.fit(train[c[0:3]], train[c[-1:]].values.ravel())
# </code></details>
# 

# In[17]:


scores_train = clf.score(train[c[0:3]], train[c[-1:]].values.ravel())
scores_test = clf.score(test[c[0:3]], test[c[-1:]].values.ravel())
print('Accuracy train DataSet: {: .1%}'.format(scores_train), 'Accuracy test DataSet: {: .1%}'.format(scores_test))
plot_confusion_matrix(clf, test[c[0:3]], test[c[-1:]].values.ravel())  
plt.show()


# As you can see, the accuracy is not bad. The difference between the training and test sets is little. It means that the model is not bad, and for increasing the accuracy we should increase our DataSet. You can test it yourself by adding all the directories with images.
# 

# # Create your own mask of land use
# 

# Let's build your own mask based on our classifier model.
# 
# First of all, you need to choose a few images from the downloaded data list and build your DataSet.
# 

# In[18]:


test_image = 8 # choose the number of images from the data list
mask_test = data[test_image:test_image+1] # Test Image + Mask
mask_test_DataSet = create_DataSet(mask_test) #Build a DataSet
print(mask_test_DataSet)


# Then, calculate the hex colour of classes using our model:
# 

# In[19]:


c = mask_test_DataSet.columns
mask_test_predict = clf.predict(mask_test_DataSet[c[0:3]])
print(mask_test_predict)


# In[20]:


size = mask_test[0][1].size #get original image size
print(size)
predict_img = np.array(mask_test_predict).reshape((size[1], size[0])) #reshaping array of HEX colour
print(predict_img)


# In[22]:


rgb_size = np.array(mask_test[0][0]).shape
print("Image size: ", rgb_size)
predict_img_rgb = np.zeros(rgb_size)
for i, r in enumerate(predict_img):
    for j, c in enumerate(r):
        predict_img_rgb[i, j, 0], predict_img_rgb[i, j, 1], predict_img_rgb[i, j, 2] = hex2rgb(c)


# Let's compare our mask with the original one.
# 

# In[23]:


predict_img_rgb = predict_img_rgb.astype('int')
print("Model mask")
plt.imshow(predict_img_rgb)
plt.show()
print("Real mask")
plt.imshow(mask_test[0][1])
plt.show()


# You can see that these masks are very similar and you just have to increase the DataSets to improve the accuracy.
# 

# ## Conclusions
# 

# In this lab, we have studied how to create an expert system based on classifiers, which allows to obtain the classification of land, based on aero and satellite images. These principles can be used for any type of images and any types of land use.
# 
# In this lab, we have learned to upload and convert images. We have mastered extracting image colors and building / training / testing sets of classifiers. We have calculated the accuracy and studied how to build our own mask based on a classifier.
# 
