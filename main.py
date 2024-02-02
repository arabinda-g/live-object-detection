import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap

# Load the TensorFlow model
model = tf.saved_model.load('ssd_mobilenet_v1_coco_2018_01_28/saved_model')
print("Model's input signature:")
print(model.signatures['serving_default'].structured_input_signature)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 800, 600)

        # Set a layout for the main window to control margins
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(10, 10, 10, 10)  # Margins: left, top, right, bottom

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def resizeEvent(self, event):
        # This method is called whenever the window is resized.
        QMainWindow.resizeEvent(self, event)
        # Resize the QLabel to fill the window while respecting the margins
        # self.image_label.resize(self.central_widget.width(), self.central_widget.height())

        
        # Resize the image_label while maintaining aspect ratio
        scaled_size = self.central_widget.size() - QSize(20, 20)  # Subtract margins
        self.image_label.setFixedSize(scaled_size)

        # You may also need to adjust the scaling of the displayed image here
        # depending on how you're updating the QLabel with the video frames.

    def update_frame(self):
        ret, self.image = self.capture.read()

        # Flip the image horizontally
        # self.image = cv2.flip(self.image, 1)

        if ret:
            # Object detection
            self.image = self.detect_objects(self.image)

            # Convert to Qt format
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            step = channel * width
            # q_img = QImage(image.data, width, height, step, QImage.Format_RGB888)

            # self.image_label.setPixmap(QPixmap.fromImage(q_img))
            # Assuming you have a frame to display, resize it to fit the label while maintaining aspect ratio
            # Note: cv2.resize might distort the aspect ratio, so consider using Qt's scaling methods
            qt_image = QImage(image.data, image.shape[1], image.shape[0], 
                            image.strides[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def detect_objects(self, image):
        # Convert the image to RGB (OpenCV uses BGR) and then to uint8
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor([rgb_image], dtype=tf.uint8)

        # Get the callable function from the model's signature
        detect_fn = model.signatures['serving_default']

        # Prepare the input in the correct format
        input_dict = {'inputs': input_tensor}

        # Run detection
        detections = detect_fn(**input_dict)

        # Process the detections
        num_detections = len(detections['num_detections'])
        # print(f"Detected {num_detections} objects")
        for i in range(num_detections):  # Iterate over all possible detections
            score = detections['detection_scores'][i].numpy()[0]
            # if score < 0.5:  # Skip detections with a low score
            #     continue

            # Extract the bounding box coordinates
            # box = detections['detection_boxes'][i].numpy()[:4]  # Adjusted to handle multi-dimensional box
            # ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            all_boxes = detections['detection_boxes'][i].numpy()

            # Filter out the boxes with low confidence
            all_boxes= all_boxes[np.all(all_boxes > 0, axis=1)]

                # Get the class ID and label
            class_id = int(detections['detection_classes'][i].numpy()[0])
            class_name = self.get_class_name(class_id)

            # image = self.draw_box(image, box, class_name, score)

            # Loop through box
            for j in range(len(all_boxes)):
                # if box[j] > 0:
                image = self.draw_box(image, all_boxes[j], class_name, score)


        return image

    def draw_box(self, image, box, class_name, score):
        # Draw a bounding box and label on the image
        h, w, _ = image.shape
        ymin, xmin, ymax, xmax = box
        start_point = (int(xmin * w), int(ymin * h))
        end_point = (int(xmax * w), int(ymax * h))

        # Green color for the box border
        box_color = (0, 255, 0)  # Green color
        # White color for the label background
        label_bg_color = (255, 255, 255)  # White color
        # Black color for the text
        text_color = (0, 0, 0)  # Black color
        box_thickness = 2
        image = cv2.rectangle(image, start_point, end_point, box_color, box_thickness)

        # Define the font for the label
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Smaller font size
        font_scale = 0.4
        font_thickness = 1

        # Calculate the size of the label and create a filled rectangle as the background
        label = f"{class_name}: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_background_start = (start_point[0], start_point[1] - label_height - baseline - 5)
        label_background_end = (start_point[0] + label_width, start_point[1])
        image = cv2.rectangle(image, label_background_start, label_background_end, label_bg_color, thickness=cv2.FILLED)

        # Put the label text on top of the background
        label_offset = (start_point[0], start_point[1] - 5)
        image = cv2.putText(image, label, label_offset, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return image

    def get_class_name(self, class_id):
        class_names = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
            22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
            28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
            35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
            40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
            44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
            57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
            63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
            72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
            78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
            85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }  # Complete this based on the model's dataset
        return class_names.get(class_id, "Unknown")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
