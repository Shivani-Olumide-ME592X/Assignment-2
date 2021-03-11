import cv2
import cv2 as cv
import numpy as np
def mask_grid_lines(original_image,coordinates):
    images=original_image[coordinates[0][0]:coordinates[1][0],coordinates[0][1]:coordinates[1][1]]
    image_hsv=cv2.cvtColor(images,cv2.COLOR_BGR2HSV)
    ret,image_result = cv2.threshold(image_hsv[:,:,2],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours= cv2.findContours(image_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    c=sorted(contours,key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(c)
    img=images[:y,:]
    img_coordinate=y
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1]
    s = cv2.Laplacian(img_hsv, cv2.CV_8UC1, ksize=3)
    s=cv2.convertScaleAbs(s,alpha=1.5)
    bw=s.copy()
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // 25
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    i_r = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel, )
    ret,image_result = cv2.threshold(i_r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)   
    contours= cv2.findContours(image_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    image_copy=img.copy()
    horizontal_coordinates=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w>50):
            horizontal_coordinates.append(y)
    horizontal_coordinates=list(set(horizontal_coordinates))        
    horizontal_coordinates=sorted(horizontal_coordinates)
    if(horizontal_coordinates[1]-horizontal_coordinates[0]<=10):
        horizontal_coordinates.remove(horizontal_coordinates[0])
    
    if(horizontal_coordinates[-1]-horizontal_coordinates[-2]<=10):
        horizontal_coordinates.remove(horizontal_coordinates[-1])
    diff=[]
    for i in range(len(horizontal_coordinates)-1):
        if(horizontal_coordinates[i+1]-horizontal_coordinates[i]>35):
            diff.append(horizontal_coordinates[i+1]-horizontal_coordinates[i])            
    width=max(set(diff), key = diff.count) 
    verticalsize = width
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    i_r = cv2.morphologyEx(vertical, cv2.MORPH_DILATE, kernel, )
    i_r=cv2.convertScaleAbs(i_r,alpha=1.5)
    ret,image_result = cv2.threshold(i_r,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours= cv2.findContours(image_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    ht=[]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)    
        if(h>50):
            ht.append(x)    
    ht=sorted(list(set(ht)))
    count=[]
    elts=[]
    for i in range(1,int(np.ceil(len(ht)/2))):
        cnt=0
        e=[]
        for j in range(i+1,len(ht)-1):
            difference=ht[j]-ht[i]
            if(difference>=42):
                if((ht[j]-ht[i])%width<=6):
                    cnt+=1
                    e.append(ht[j])
        e.append(ht[i])
        elts.append(e)        
        count.append(cnt)
    vertical_coordinates=elts[count.index(max(count))]
    vertical_coordinates=sorted(vertical_coordinates)
    extreme=[ht[0],ht[-1],0,images.shape[1]]
    for i in extreme:
        if (i not in vertical_coordinates):
            vertical_coordinates.append(i)
    e1,e2,e3,e4=0,0,0,0
    for i in range(len(vertical_coordinates)):
        if(vertical_coordinates[i]>5 and horizontal_coordinates[i]<25):
            e1= vertical_coordinates[i]
            break   
    for i in range(len(vertical_coordinates)):
        if(images.shape[1]-vertical_coordinates[i]<20):
            e2= vertical_coordinates[i]
            break 
    horizontal_coordinates=sorted(horizontal_coordinates)
    extreme=[0,images.shape[0]]
    for i in extreme:
        if (i not in horizontal_coordinates):
            horizontal_coordinates.append(i)
    for i in range(len(horizontal_coordinates)):
        if(horizontal_coordinates[i]>5 and horizontal_coordinates[i]<50):
            e3= horizontal_coordinates[i]
            break
    for i in range(len(horizontal_coordinates)):
        if(images.shape[0]-horizontal_coordinates[i]<20):
            e4= horizontal_coordinates[i]
            break 
    horizontal_coordinates=sorted(horizontal_coordinates)    
    vertical_coordinates=sorted(vertical_coordinates)    
    def unique_count_app(a):
        colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
        return colors[count.argmax()]
    r_c,g_c,b_c=unique_count_app(img)
    color = ( int (r_c), int (g_c), int (b_c))
    for i in horizontal_coordinates:
        if(i>=e3 and i<=e4):
            image_copy=cv2.line(image_copy,(0,i),(img.shape[1],i),color,2)
    for i in vertical_coordinates:
        if(i>=e1 and i<=e2):
            image_copy=cv2.line(image_copy,(i,0),(i,img.shape[0]),color,2)
    original_image=np.zeros_like(original_image)
    original_image[coordinates[0][0]:coordinates[0][0]+img_coordinate,coordinates[0][1]:coordinates[1][1]]=image_copy
    return original_image
