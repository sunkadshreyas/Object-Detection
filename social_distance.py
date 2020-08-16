import numpy as np
import cv2
import time
from scipy.spatial import distance as dist

"""
Start of:
Reading input video
"""
VIDEO_PATH = 'videos/walking.mp4'
video = cv2.VideoCapture(VIDEO_PATH)

writer = None
h, w = None, None

"""
End of:
Reading input video
"""


"""
Start of:
Loading YOLO v3 network
"""

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg', 'yolo-coco-data/yolov3.weights')

layers_names_all = network.getLayerNames()

layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5
threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""
f = 0
t = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    if w is None or h is None:
        h, w = frame.shape[:2]

    """
    Start of:
    Getting blob from current frame
    """
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """
    network.setInput(blob)  
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    f += 1
    t += end - start
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """
    bounding_boxes = []
    confidences = []
    class_numbers = []
    centroids = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if class_current == 0 and confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min,int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
                centroids.append((x_center, y_center))

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """
    final_results = []
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()
            
            r = (confidences[i],(x_min,y_min,x_min+box_width,y_min+box_height),centroids[i])
            final_results.append(r)
            
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    End of:
    Drawing bounding boxes and labels
    """
    """
    Checking for distance between people
    """
    violations =set()
    
    if len(final_results) >=2 :
        
        centroids = np.array([r[2] for r in final_results])
        D = dist.cdist(centroids, centroids, metric='euclidean')
        
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                
                if D[i,j] < 100:
                    violations.add(i)
                    violations.add(j)
                    
                    
    for (i, (prob,boundbox,centroid)) in enumerate(final_results):
        (x_start, y_start, x_end, y_end) = boundbox
        (c_x,c_y) = centroid
        color = (0,255,0)
        
        if i in violations:
            color = (0,0,255)
        
        cv2.rectangle(frame, (x_start,y_start), (x_end,y_end), color,10)
        cv2.circle(frame, (int(c_x),int(c_y)), 5, color, 1)
               
    text = "Violations : {}".format(len(violations))
    cv2.putText(frame, text, (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                4.0,
                (0,0,255),5)
    """
    Start of:
    Writing processed frame into the file
    """
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter('videos/result-walking.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    
    """
    End of:
    Writing processed frame into the file
    """

"""
End of:
Reading frames in the loop
"""


# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()

