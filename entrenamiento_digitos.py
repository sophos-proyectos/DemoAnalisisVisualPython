# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 18:45:09 2020

@author: Usuario
"""




# Importo librerias
from skimage.feature import hog
import numpy as np
import cv2


#Se carga base de datos
from scipy.io import loadmat
print("Se carga base de datos...")
dataset  = loadmat('mnist-original.mat')

#Se separan X y Y
features = np.array(dataset['data'].T) 
labels = np.array(dataset['label'].T, 'int')


#Ejemplo para visualizar una imagen
a=features[69000].reshape((28, 28))
cv2.imshow("imagen_muestra",a)
cv2.waitKey()


list_hog_fd = []
print("Pre-proceso de datos...")
for feature in features:
    #HISTOGRAMA DE GRADIENTES ORIENTADOS
    fd,hog_image = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),visualize=True)
    list_hog_fd.append(fd)
#Se ordena
hog_features = np.array(list_hog_fd)

cv2.imshow("imagen_muestra",hog_image)
cv2.waitKey()


import numpy as np
from sklearn.model_selection import train_test_split

#Se ordentan base de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(hog_features,labels, test_size = 0.30, random_state = 100) 


#Se crea y entrena modelo
print("Entrenando...")
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
print("...")

#Guardo modelo en directorio 
print("guardadndo modelo")
import joblib
joblib.dump(clf, 'Modelo.joblib') 
#eje = joblib.load('Modelo.joblib')

#Funcion para disminuir la dimension de una lista 
def disminuir(lista):
    return([val for sublist in lista for val in sublist])

#Se verifica precision
print("Precisión:")
pred_y=clf.predict(X_test)

print((np.mean(pred_y == disminuir(y_test)))*100,"%")
y_test_1=[val for sublist in y_test for val in sublist]








# np.array(y_test)
# im = cv2.imread("MARIO.png")
# roi = cv2.resize(im, (100, 100))
# im_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# fd,hog_image  = hog(im_gray, orientations=9, pixels_per_cell=(20, 20), cells_per_block=(1, 1), visualize=True)
# roi = cv2.resize(fd, (127, 127))
# fd,hog_image = hog(feature.reshape((28, 28)), orientations=11, pixels_per_cell=(7, 7), cells_per_block=(1, 1),visualize=True)
# cv2.imshow("imagen_muestra",im)
# cv2.waitKey()

