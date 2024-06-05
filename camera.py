import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)  # Abre a câmera padrão (0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)  # Atualiza o frame a cada 20 ms

    def initUI(self):
        self.setWindowTitle("Feed da Câmera")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.capture_button = QPushButton("Capturar", self)
        self.capture_button.clicked.connect(self.capture_image)
        self.layout.addWidget(self.capture_button)

        self.captured_label = QLabel(self)
        self.layout.addWidget(self.captured_label)

        self.setLayout(self.layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(image))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imwrite("captured_image_28x28.jpg", resized)
            print("Imagem capturada e salva como 'captured_image_28x28.jpg'")

            # Converta a imagem redimensionada para exibir na interface
            resized_image = QImage(resized.data, resized.shape[1], resized.shape[0], resized.strides[0], QImage.Format_Grayscale8)
            self.captured_label.setPixmap(QPixmap.fromImage(resized_image))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
