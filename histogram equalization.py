import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Pepper.tiff",0)
#cv2.imshow('pepper.tiff',img)
#plt.hist(img.ravel(),256,[0,256])
m=img.shape[0]
n=img.shape[1]
grayImg = np.zeros((img.shape[0], img.shape[1]))
for i in range(m):
    for j in range(n):
        grayImg[i][j] = round(img[i][j])

unique, value = np.unique(img, return_counts=True)
counts = dict(zip(unique,value))
L = len(counts)
size=m*n
pdf = [counts[i]/size for i in counts]

sum=0
cdf = []
for i in range(len(pdf)):
    sum =sum+ pdf[i]
    cdf.append(sum)

newCdf = [int(i*(L - 1)) for i in cdf]
newCdfDict = {}
for i in range(len(newCdf)):
    newCdfDict[i] = newCdf[i]

newImg = np.zeros(grayImg.shape)
for i in range(grayImg.shape[0]):
    for j in range(grayImg.shape[1]):
        try:
            newImg[i][j] = newCdfDict[grayImg[i][j]]
        except:
            continue
plt.hist(newImg.ravel(),256,[0,256])
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2,2,1)
ax1.set_title('Image')
ax1.imshow(img,cmap='gray')
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('Equalized Image')
ax2.imshow(newImg,cmap='gray')

plt.show()
