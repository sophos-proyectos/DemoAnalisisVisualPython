

# Importan librerias
import joblib
from skimage.feature import hog
import numpy as np
import cv2



#Se carga el modelo 
eje = joblib.load('Modelo.joblib')



# lee la imagen
im = cv2.imread("2.jpg")


#Escala de grises
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)



roi = cv2.resize(im_th, (28, 28))#, interpola

#HISTOGRAMA DE GRADIENTES ORIENTADOS
roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
#Se predice
nbr = eje.predict(np.array([roi_hog_fd]))
#Se verifica tamaño para posicion de prediccion
height, width, channels = im.shape
#Se escribe la prediccion en la imagen
cv2.putText(im, str(int(nbr[0])), (int(height/4), int(width/2)),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
#Se imprime la imagen
cv2.imshow("imagen_muestra",im)
cv2.waitKey()
