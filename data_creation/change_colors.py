import cv2
import numpy as np
import json
import sys
from copy import deepcopy


def change_color(img_path, res_path):
  img = cv2.imread(img_path)

  # convert to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv)

  red_hue = 120
  blue_hue = 54

  # diff hue (red_hue - blue_hue)
  diff_hue = red_hue - blue_hue

  # create mask for blue color in hsv
  upper = (255,	255, 240)
  lower = (80,	80,	0)
  mask = cv2.inRange(hsv, lower, upper)
  mask = cv2.merge([mask,mask,mask])

  # apply morphology to clean mask
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # modify hue channel by adding difference and modulo 180
  hnew = np.mod(h + diff_hue, 180).astype(np.uint8)

  # recombine channels
  hsv_new = cv2.merge([hnew,s,v])

  # convert back to bgr
  bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

  # blend with original using mask
  result = np.where(mask==(255, 255, 255), bgr_new, img)
  cv2.imwrite(res_path, result)
