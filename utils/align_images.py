from __future__ import print_function
import cv2
import numpy as np
import os
import shutil


MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.1


def alignImage(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
  matches = matcher.match(descriptors1, descriptors2, None)
  # matches = matcher.knnMatch(descriptors1, descriptors2, 2)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  # matches = matches[:4]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h


def alignImages(refFolder, png_folder):  

  folders = os.listdir(refFolder)
  folders.sort()
  folders = folders[:-3]
  png_files = os.listdir(png_folder)
  png_files.sort()

  for i in range(len(folders)):

    folder_path = refFolder + folders[i] + "/"

    # Read reference image
    refFilename = folder_path + "container.png"

    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    # imReference = cv2.resize(imReference, (384*2, 384*2))

    # Read image to be aligned
    imFilename = png_folder + "/" + png_files[i]
    print("Reading image to align : ", imFilename);  
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    h, w, _ = im.shape
    im = cv2.resize(im, (w // 6, h // 6))

    print("Aligning images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    imReg, h = alignImage(im, imReference)
    
    # Write aligned image to disk. 
    outFilename = folder_path + "warped_screenshot.png"
    print("Saving aligned image : ", outFilename); 
    # imReg = cv2.resize(imReg, (128*3, 128*3))
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n",  h)

def deleteFiles(folder):
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))