import sys
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import cv2

from easydict import EasyDict as edict
from simplejson import loads
import json
import urllib

import os
#import base64
from io import BytesIO
from os import listdir
from os.path import isfile, join
import glob
import requests

import time
import csv
import logging
import imutils


image_file_path = 'sample/d06p1.jpg'

image = cv2.imread(image_file_path)
#cv2.imshow("Input:",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 10, 50)
cnts,hier = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
screenCnt = None
'''
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
'''
largestContour = 0
for i in range(1, len(cnts)):
    if cv2.contourArea(cnts[i]) > cv2.contourArea(cnts[largestContour]):
        largestContour = i
screenCnt = cnts[largestContour]
#cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
if screenCnt is None:
    print("contour is none")
else:
    print("contour is %d", largestContour)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

resized_image = cv2.resize(image,(image.shape[1]//3,image.shape[0]//3),interpolation=cv2.INTER_CUBIC)
cv2.imshow("Contour:",resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()