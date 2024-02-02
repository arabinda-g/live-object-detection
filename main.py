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
        self.image = cv2.flip(self.image, 1)

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
        for i in range(len(detections['detection_scores'])):  # Iterate over all possible detections
            score = detections['detection_scores'][i].numpy()[0]
            if score < 0.5:  # Skip detections with a low score
                continue

            box = detections['detection_boxes'][i].numpy()[:4]  # Adjusted to handle multi-dimensional box
            class_id = int(detections['detection_classes'][i].numpy()[0])
            class_name = self.get_class_name(class_id)

            image = self.draw_box(image, box, class_name, score)

        return image

    def draw_box(self, image, box, class_name, score):
        # Assuming box is a 2D array with the required values in the first row
        ymin, xmin, ymax, xmax = box[0]
        h, w, _ = image.shape
        start_point = (int(xmin * w), int(ymin * h))
        end_point = (int(xmax * w), int(ymax * h))
        # color = (0, 255, 0)  # Green color for the box
        # red color for the box
        color = (0, 0, 255)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        # Put a label near the box
        label = f"{class_name}: {score:.2f}"
        image = cv2.putText(image, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2, cv2.LINE_AA)
        return image

    def get_class_name(self, class_id):
        class_names = {1: "person", 2: "bicycle", 3: "car",}  # Complete this based on the model's dataset
        return class_names.get(class_id, "Unknown")




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
