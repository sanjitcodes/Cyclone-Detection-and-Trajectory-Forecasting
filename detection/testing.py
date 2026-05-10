import cv2
import numpy as np

net=cv2.dnn.readNet("Models\yolo_training_best.weights", "Models\yolo_testing_best.cfg")

classes=['cyclone']

layer_names=net.getLayerNames()

outputlayers= [layer_names[i-1] for i in net.getUnconnectedOutLayers() ]

colors= np.random.uniform(0,255,size=(len(classes),3 ))

img=cv2.imread(r"68.jpg")

height,width,channels = img.shape

blob=cv2.dnn.blobFromImage(img,0.00392, (416,416) , (0,0,0), True, crop=False)




net.setInput(blob)
outs=net.forward(outputlayers)

boxes=[]
confidences=[]
class_ids=[]

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]

        if confidence>0.2:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h=int(detection[3]* height)

            x=int(center_x - w /2)
            y=int(center_y - h /2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes , confidences , 0.5, 0.4 )
font=cv2.FONT_HERSHEY_PLAIN
for i in range (len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label=classes[class_ids[i]]
        confidence = confidences[i]
        color=colors[class_ids[i]]
        cv2.rectangle(img , (x,y) ,(x+w , y+h) ,(0,255,255), 2)
        cv2.putText(img,label+ " "+ str(round(confidence, 2)) , (x,y+30) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,0,255) ,2)

cv2.imwrite('img/4.jpg',img)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

