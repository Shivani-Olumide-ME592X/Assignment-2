import cv2
import numpy as np
def extraction(segment,original_image):
    image=segment[20:-20,20:-20]
    image_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    ret,image_result = cv2.threshold(image_hsv[:,:,1],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours= cv2.findContours(image_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    mask=np.zeros_like(image_result)
    for c in contours:
        if(cv2.contourArea(c)<400):
                x,y,w,h = cv2.boundingRect(c)
                cv2.drawContours(mask,contours,-1,(255,255,255), 2) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, )
    image_result=np.zeros_like((segment[:,:,0]))
    image_result[20:-20,20:-20]=mask
    crop1=cv2.bitwise_and(original_image,original_image,mask=np.uint8(image_result))
    image_lab=cv2.cvtColor(crop1,cv2.COLOR_BGR2LAB)
    ret,image_result = cv2.threshold(image_lab[:,:,0],-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_CLOSE, kernel, )
    contours= cv2.findContours(image_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    crop2=cv2.bitwise_and(original_image,original_image,mask=np.uint8(image_result))
    if(len(np.where(crop2==0))/crop2.shape[0]*crop2.shape[1]*crop2.shape[2]>0.98):
        return crop1
    else:
        return crop2
