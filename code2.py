import cv2
import numpy as np

def count_people_on_image(image_path):
    # Load YOLOv3 COCO names
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Load YOLOv3 model
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Get the names of the YOLO output layers
    output_layer_names = net.getUnconnectedOutLayersNames()
    
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return
    
    # Perform forward pass through the network to get the detections
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)
    
    # Process the detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for confidence
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and display/save the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # save the image with bounding boxes
    cv2.imwrite("output_image_with_boxes.jpg", image)
  
    # Count people
    num_people = 0
    for i in range(len(boxes)):
        if i in indexes:
            label = classes[class_ids[i]]
            if label == 'person':
                num_people += 1
    
    # Print the count of people
    print(f"Number of People: {num_people}")

if __name__ == "__main__":
    image_path = 'C:\\Users\\shrey\\Desktop\\blog-ppl counter\\image2.jpg'
    count_people_on_image(image_path)
