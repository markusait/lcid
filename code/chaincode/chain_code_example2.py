import cv2
import numpy as np
import sys

sys.setrecursionlimit(100000)

def region_grow (i,j,g_min,g_max,dir): 
    if array_img[i,j]>g_min and array_img[i,j]<g_max and labelled[i,j]==0:
        direction.append(dir)
        labelled [i,j]=1
        if j<width_img-1 and j>1 and i<hight_img-1 and i>1:
            region_grow(i-1,j,g_min,g_max,1)
            region_grow(i,j+1,g_min,g_max,3)
            region_grow(i+1,j,g_min,g_max,5)
            region_grow(i,j-1,g_min,g_max,7)
            region_grow(i+1,j+1,g_min,g_max,4)
            region_grow(i-1,j-1,g_min,g_max,8)
            region_grow(i+1,j-1,g_min,g_max,6)
            region_grow(i-1,j+1,g_min,g_max,2)
                                       
def chain (image): 
    image [image == 255] = 1
    h_img,w_img=image.shape
    chain_array=np.array([8,1,2,7,0,3,6,5,4])
    chain_dir=np.array ([(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)])
    chain_dir1=np.array([(-1,0),(0,-1),(0,1),(-1,0),(0,0),(-1,0),(0,-1),(0,-1),(1,1)])
    chain_dir2=np.array([(0,-1),(0,1),(-1,0),(1,0),(0,0),(1,0),(0,1),(0,1),(1,0)])
        
    # getting starting point for the chain-code
    for x1 in range (2,w_img-2):
        for y1 in range (2,h_img-1):
            if image[y1,x1] != 1:
                a = image [y1-1,x1] + image [y1+1,x1] + image [y1,x1-1] + image [y1,x1+1] + image [y1-1,x1+1] + image [y1+1,x1+1] + image [y1-1,x1-1] + image [y1+1,x1-1]               
                image_chain [y1,x1] = a
    for x1 in range (2,w_img-2):
        for y1 in range (2,h_img-1):
            if image[y1,x1] == 0:
                spx = x1
                spy = y1
                break
        if spx>0 or spy>0:
            break      
      
    # getting chain code from labelled (region_grow) and smoothed (CV2.findContours) image
    sumarray_max=np.amax(image_chain[spy-1:spy+2, spx-1:spx+2])
    chain_list=np.array([h_img,w_img,spy,spx,0])
    while np.sum(image_chain) > 0 and sumarray_max > 0:
        image_chain[spy,spx]=0
        b=np.argmax(image_chain[spy-1:spy+2, spx-1:spx+2])
        sumarray_max=np.amax(image_chain[spy-1:spy+2, spx-1:spx+2])
        chain_list=np.append(chain_list,chain_array[b])
        y1,x1=chain_dir[b] 
        y2,x2=chain_dir1[b]
        y3,x3=chain_dir2[b]   
        image_chain[spy+y2,spx+x2]=0
        image_chain[spy+y3,spx+x3]=0
        spx=spx+x1
        spy=spy+y1
    return chain_list

def draw_chain (canvas_y, canvas_x, starty, startx, code, line):
    chain_dir=np.array ([(0,0),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)])
    canvas=np.zeros(shape=[canvas_y, canvas_x, 3], dtype=np.uint8)
    for i in code[2:]:
        y,x=chain_dir[i]
        y=y+starty
        x=x+startx
        canvas=cv2.line(canvas,(startx,starty),(x,y),(0,255,0),1)
        starty=y
        startx=x        
    return canvas

#def draw_roi(event,x,y,flags,param):
#    global ix,iy,roi, img_test, img_start,buffer
#    if event == cv2.EVENT_LBUTTONDOWN:
#        img_test=img_start[y:y+120,x:x+180]
#        roi=1        
#    elif event == cv2.EVENT_MOUSEMOVE:
#        if y+120 < h_img and x+180 < w_img:
#            cv2.rectangle(img_start,(x,y),(x+180,y+120), (0,255,0),2)
#            cv2.circle(img_start,(x+90,y+60),20,(0,255,255),2)
#            cv2.line(img_start,(x+90,y+30),(x+90,y+90),(0,255,0),2)
#            cv2.line(img_start,(x+60,y+60),(x+120,y+60),(0,255,0),2)
#            cv2.imshow(windowName,img_start)
#            img_start=np.copy(img)
#          
# get the "region of Interest" (ROI)" by draw_roi()
# it is necessary to left-click first, followed by pressing a key on keyboard
# function has to be rewritten, because of insufficent method of moving the box 

#windowName="Sonobild"
img = cv2.imread("BN-1.bmp",0)
#img_start = cv2.imread("BN-1.bmp",0)

#h_img, w_img=img_start.shape
h_img, w_img=(0,407)
array_img=img[h_img:h_img+120,w_img:w_img+180]

#roi=0
#cv2.namedWindow(windowName)
#cv2.imshow(windowName, img_start)
#cv2.setMouseCallback(windowName, draw_roi)
#cv2.imshow(windowName, img_start)
#while roi==0:
#    cv2.waitKey(0)
#cv2.destroyAllWindows()
#array_img=img_test

cv2.namedWindow("ROI")
cv2.moveWindow("ROI",0,0)
cv2.imshow ("ROI",array_img)

# draw the original image from comparision to segementation steps
cv2.namedWindow("Orginal Image")
cv2.moveWindow("Orginal Image",480,0)
cv2.imshow ("Orginal Image",img)


# get the mean value and the standard deviation of the central region of the ROI
hight_img,width_img=array_img.shape
labelled=np.zeros (array_img.shape)
start_point_y=round(hight_img/2)
start_point_x=round(width_img/2)
mean=np.mean(array_img[start_point_y-20:start_point_y+20,start_point_x-10:start_point_x+10])
std=np.std(array_img[start_point_y-10:start_point_y+10,start_point_x-10:start_point_x+10])
upper_limit=mean+std
if mean-2*std>0:
    lower_limit=mean-2*std
elif mean-std<0:
    lower_limit=0

# image segmentation step 1 by region growing using region_grow ()
contours=np.copy(array_img)
contours.fill(0)
direction=[120,180,start_point_y,start_point_x,0]
while np.sum(labelled)==0:
    region_grow (start_point_y,start_point_x,lower_limit,upper_limit,0)
    np.sum(labelled)
    start_point_y=start_point_y-1        
cv2.namedWindow("Region filling")
cv2.moveWindow("Region filling",0,210)
cv2.imshow ("Region filling",labelled)

# image segmentation step 2 by "filling the holes" resulting in moderatly smoothing the surface 
npMask=np.array(labelled,dtype="i8")
_,labelled_holes_filled, hierarchy = cv2.findContours(npMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours, labelled_holes_filled, -1, (255, 0, 0 ),10)
contours = ~contours
cv2.namedWindow("Filling the holes")
cv2.moveWindow("Filling the holes",0,400)
cv2.imshow ("Filling the holes",contours)

merged=cv2.add(array_img,contours)

cv2.namedWindow("Extracted liver segment")
cv2.moveWindow("Extracted liver segment",0,600)
cv2.imshow ("Extracted liver segment",merged)

# get chain-code of surface using chain () and draw chain-code by draw_chain ()
image_chain=np.zeros((120,180))
chain_code=chain (contours)
surface_chain=draw_chain (120,180,32,2,chain_code[3:],1)
cv2.namedWindow("chain_code")
cv2.moveWindow("chain_code",1100,0)
cv2.imshow ("chain_code",surface_chain)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print chain code of surface from chain () and tissue from region_grow ()
print ("chain code of surface: ",chain_code)
print ("chain code of tissue: ", direction)