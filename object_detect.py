"""
Object Detection in Images using YOLO

Algorithm : 
    Step 1 = Reading RGB Image
    Step 2 = Getting Blob
    Step 3 = Loading YOLO Network
    Step 4 = Forward Propagation
    Step 5 = Getting bounding boxes
    Step 6 = Non Max Supression
    Step 7 = Drawing bounding boxes with labels
"""

import numpy as np
import cv2
import time

"""
Step 1 : Reading RGB Image
"""
image_path = ''

image = cv2.imread(image_path)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image',image)

cv2.waitKey(0)

cv2.destroyWindow('Original Image')

print('Image Shape : ', image.shape)
# (height, width, number_of_channels)

height,width = image.shape[:2]

"""
Step 2 : Getting Blob from Image
"""

blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB=True, crop=False)
# Returns 4 dimensional blob 
# (number_of_images, number_of_channels, height, width)

print('Blob Shape : ',blob.shape)

# Displaying Blob
display_blob = blob[0,:,:,:].transpose(1,2,0)
print('Display Blob Shape : ',display_blob.shape)
# (height, width, number_of_channels)

cv2.namedWindow('Blob Image',cv2.WINDOW_NORMAL)
cv2.imshow('Blob Image', cv2.cvtColor(display_blob, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)

cv2.destroyWindow('Blob Image')

"""
Step 3 : Loading YOLO Network
"""

# The various named of classes is in coco.names
with open('yolo-coco-data/coco.names') as file:
    labels = [line.strip() for line in file]
    
# The various class names
print('The number of classes', len(labels))
# print('The various classes : ',labels)   

# Load the network
cfg_path = 'yolo-coco-data/yolov3.cfg'
weights_path = 'yolo-coco-data/yolov3.weights'
network = cv2.dnn.readNetFromDarknet(cfg_path, weights_path) 

# getting all the layers
all_layers = network.getLayerNames()
# print('All Layer Names : \n',all_layers)

# output layers
output_layers = \
    [all_layers[i[0] - 1] for i in network.getUnconnectedOutLayers()]
print('Output layer names : \n', output_layers)

minimum_probability = 0.5
nonmax_threshold = 0.3

# generate colors for bounding boxes of all labels
colors = np.random.randint(0, 255, size=(len(labels),3), dtype='uint8')
print('The number of colors required : ',colors.shape)
print('Color of bounding box for label 0 : ',colors[0])

"""
Step 4 : Forward Propagation
"""
    
# Implementing Forward Propagation

network.setInput(blob)
start = time.time()
output_from_network = network.forward(output_layers)
end = time.time()

print('Forward propagation took {:.3f} seconds'.format(end-start))

"""
Step 5 : Getting Bounding Boxes
"""
# output consists of 85 values 
# first 5 values = coordinates and object exists or not
# next 80 values = probability of each object 

# list of coordinates of rectangle for each object
bounding_boxes = []
# probability that object exists
confidences = []
# which object it is out of 80 classes
class_numbers = []

for result in output_from_network:
    for detected_objects in result:
        
        scores = detected_objects[5:]
        class_max_confidence = np.argmax(scores)
        confidence_of_class = scores[class_max_confidence]
        
        # print('Shape of detected objects : ',detected_objects.shape)
        
        if confidence_of_class > minimum_probability:
            
            box_shape = detected_objects[0:4] * np.array([width,height,width,height])
            x_centre, y_centre, box_width, box_height = box_shape
            x_min = int(x_centre - (box_width/2))
            y_min = int(y_centre - (box_height/2))

            bounding_boxes.append([x_min,y_min,int(box_width),int(box_height)])
            confidences.append(float(confidence_of_class))
            class_numbers.append(class_max_confidence)  


"""
Step 6 : Non Max Supression
"""                  

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, minimum_probability,nonmax_threshold)

"""
Step 7 : Drawing the Bounding Boxes
"""

objects = 0

if len(results) > 0:
    
    for i in results.flatten():
        
        print('Object {0} : {1}'.format(objects, labels[int(class_numbers[i])]))
        
        objects += 1
        
        x_min = bounding_boxes[i][0]
        y_min = bounding_boxes[i][1]
        box_width = bounding_boxes[i][2]
        box_height = bounding_boxes[i][3]
        
        current_box_color = colors[class_numbers[i]].tolist()
        
        cv2.rectangle(image, 
                      (x_min,y_min), 
                      (x_min+box_width, y_min+box_height),
                      current_box_color,2)
        
        current_box_text = '{} : {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])
        
        cv2.putText(image, current_box_text, (x_min, y_min-5), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, current_box_color, 2)
        
        
print('Total objects detected : ',len(bounding_boxes))
print('Objects remaining after Non Max Supression : ',objects)

cv2.namedWindow('Detections',cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyWindow('Detections')

print('\n-----\nEnd of Program\n-----\n')
"""
End of Program
"""        
            



