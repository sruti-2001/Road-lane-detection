import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img):
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey,(5,5),0)

    #canny : It will perform a derivative and measure the adjacent changes in intensity in all directions, x and y
    canny=cv2.Canny(blur, 50,150)
    return canny

def region_of_interest(img):
    height=img.shape[0]
    poly=np.array([[(50,height),(600,height),(325,205)]])
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,poly,255)
    masked_image= cv2.bitwise_and(img,mask)
    return masked_image

def display_lines(img,lines):
    line_img=np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_img,(x1,y1),(x2,y2),(255,0,0),10)
    return line_img

def make_coordinates(img,line_parameters):
    slope ,intercept=line_parameters
    y1=img.shape[0]
    y2=int(y1*(3.5/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intersect(img,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parametes=np.polyfit((x1,x2),(y1,y2),1)
        slope=parametes[0]
        intercept=parametes[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit, axis=0)
    left_line=make_coordinates(img,left_fit_average)
    right_line=make_coordinates(img,right_fit_average)
    print(left_fit)
    return np.array([left_line,right_line])



cap=cv2.VideoCapture("Lane_video2.mp4")
while(cap.isOpened()):
    _, frame=cap.read()
    canny_img=canny(frame)
    cropped_img=region_of_interest(canny_img)
    lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=10)
    print(lines)
    avg_lines=average_slope_intersect(frame,lines)
    line_img=display_lines(frame,avg_lines)
    combo_img=cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow('result', combo_img)
    if cv2.waitKey(1) == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()