
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def process_image(img_rgb, thresh_low=50, thresh_high=250, eps=0.005, **kwargs):
  img_out = img_rgb.copy()
  img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
  thresh = np.logical_and(thresh_high > img_gray, img_gray > thresh_low)
  thresh = thresh.astype(np.uint8)
  contours, hierarchy = cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
  keys = sorted([(-cv.contourArea(v), i) for i, v in enumerate(contours)])

  idx = 0
  contour_idx = keys[idx][1]
  contour = contours[contour_idx]

  epsilon = eps * cv.arcLength(contour, True)
  approx = cv.approxPolyDP(contour, epsilon, True)
  hull_idx = cv.convexHull(contour, returnPoints=False)
  hull = cv.convexHull(contour, False)
  hull_idx[::-1].sort(axis=0)
  defects = cv.convexityDefects(contour, hull_idx)

  ppt = []
  if defects is not None:
    for i in range(len(defects)):
      s, e, f, d = defects[i, 0]
      far = contour[f][0]
      cv.circle(img_out, far, 5, [0, 0, 255], -1)
      ppt.append(cv.pointPolygonTest(
        hull, (float(far[0]), float(far[1])), True))

  cv.drawContours(img_out, contours, contour_idx, (255, 0, 0), 2)
  cv.drawContours(img_out, [approx], -1, (0, 255, 0), 2)
  # cv.drawContours(img_out,[box],-1,(0,0,255), 2)
  cv.drawContours(img_out, [hull], -1, (0, 255, 255), 2)

  average_defect = 0 if not ppt else np.mean(ppt)

  return average_defect, img_out, thresh


def convexity_score(image_paths, modifiers=dict()):
  defect_scores = []
  annotated_imgs = []
  masks = []
  for i, img_path in enumerate(image_paths):
    rgb_img = cv.cvtColor(
      cv.imread(img_path), cv.COLOR_BGR2RGB)
    modifier = modifiers.get(i, {})
    d, i, t = process_image(rgb_img, **modifier)
    defect_scores.append(d)
    annotated_imgs.append(i)
    masks.append(t)
  return defect_scores, annotated_imgs, masks
