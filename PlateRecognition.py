# plate-recognition.py

import cv2
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import DetectChars
import DetectPlates


SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


class MainWindow(QtWidgets.QMainWindow):
    plate_image_file = str()

    def __init__(self):
        super().__init__()
        self.centralwidget = QtWidgets.QWidget()
        self.statusbar = QtWidgets.QStatusBar()
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.Recognition = QtWidgets.QPushButton(
            self.scrollAreaWidgetContents_2)
        self.Choose = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.gridLayout = QtWidgets.QGridLayout(
            self.scrollAreaWidgetContents_2)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.setup_ui(self)
        self.Choose.clicked.connect(self.show_dialog)
        self.Recognition.clicked.connect(self.license_plate_recognition)
    # end function	

    def setup_ui(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(424, 409)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout.setObjectName("verticalLayout")
        size_policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        size_policy.setHorizontalStretch(1)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(
            self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(size_policy)
        self.scrollArea.setMinimumSize(QtCore.QSize(300, 380))
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2.setGeometry(
            QtCore.QRect(0, 0, 404, 378))
        self.scrollAreaWidgetContents_2.setObjectName(
            "scrollAreaWidgetContents_2")
        self.gridLayout.setObjectName("gridLayout")
        self.Choose.setObjectName("Choose")
        self.gridLayout.addWidget(self.Choose, 3, 0, 1, 1)
        self.Recognition.setObjectName("Recognition")
        self.gridLayout.addWidget(self.Recognition, 3, 1, 1, 1)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.scrollArea)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    # end function

    def retranslate_ui(self, main_window):
        main_window.setWindowTitle(QtCore.QCoreApplication.translate("BadGAI", "BadGAI"))
        self.Choose.setText(QtCore.QCoreApplication.translate("MainWindow", "Выбрать фото"))
        self.Recognition.setText(QtCore.QCoreApplication.translate("MainWindow", "Распознать"))
    # end function

    def show_dialog(self):
        self.plate_image_file = QtWidgets.QFileDialog.getOpenFileName(self, str("Open Image"), "/home",
                                                                      str("Image Files (*.png *.jpg *.bmp)"))
        file = ''.join(self.plate_image_file[0])
        pixmap = QtGui.QPixmap(file)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())
        self.resize(pixmap.width(), pixmap.height())
    # end function

    def license_plate_recognition(self):

        training_successful = DetectChars.load_and_train()

        if not training_successful:
            print("\nerror: KNN training was not successful\n")
            return

        file = ''.join(self.plate_image_file[0])
        img_original_scene = cv2.imread(file)

        if img_original_scene is None:
            msg = QtWidgets.QErrorMessage()
            msg.showMessage("Can't open Image from file")
            exit(-1)
            return

        list_of_possible_plates = DetectPlates.detect_plates_in_scene(
            img_original_scene)

        list_of_possible_plates = DetectChars.detect_chars_in_plates(
            list_of_possible_plates)

        if len(list_of_possible_plates) == 0:
            msg = QtWidgets.QErrorMessage()
            msg.showMessage("No license plate were detected")
        else:

            list_of_possible_plates.sort(
                key=lambda possible_plate: len(
                    possible_plate.strChars), reverse=True)

            lic_plate = list_of_possible_plates[0]

            if len(lic_plate.strChars) == 0:
                msg = QtWidgets.QErrorMessage()
                msg.showMessage("No characters were detected")
                return

        draw_rectangle_around_plate(img_original_scene, lic_plate)

        f = open('Plates.txt', 'a')
        f.write(lic_plate.strChars)
        f.close()

        write_license_plate_chars_on_image(img_original_scene, lic_plate)
        cv2.imwrite("img_original_scene.png", img_original_scene)
        pixmap = QtGui.QPixmap("img_original_scene.png")
        self.label.setPixmap(pixmap)
     # end function

def draw_rectangle_around_plate(img_original_scene, lic_plate):
    rect_points = cv2.boxPoints(lic_plate.rrLocationOfPlateInScene)

    cv2.line(
        img_original_scene, tuple(
            rect_points[0]), tuple(
            rect_points[1]), SCALAR_RED, 2)
    cv2.line(
        img_original_scene, tuple(
            rect_points[1]), tuple(
            rect_points[2]), SCALAR_RED, 2)
    cv2.line(
        img_original_scene, tuple(
            rect_points[2]), tuple(
            rect_points[3]), SCALAR_RED, 2)
    cv2.line(
        img_original_scene, tuple(
            rect_points[3]), tuple(rect_points[0]), SCALAR_RED, 2)
# end function

def write_license_plate_chars_on_image(img_original_scene, lic_plate):

    scene_height, scene_width, scene_num_channels = img_original_scene.shape
    plate_height, plate_width, plate_num_channels = lic_plate.imgPlate.shape

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(plate_height) / 30.0
    font_thickness = int(round(font_scale * 1.5))

    text_size, baseline = cv2.getTextSize(
        lic_plate.strChars, font_face, font_scale, font_thickness)

    ((plate_center_x, plate_center_y), (plate_width, plate_height),
     fltCorrectionAngleInDeg) = lic_plate.rrLocationOfPlateInScene

    plate_center_x = int(plate_center_x)
    plate_center_y = int(plate_center_y)

    center_of_text_area_x = int(plate_center_x)

    if plate_center_y < (scene_height * 0.75):
        center_of_text_area_y = int(round(plate_center_y)) + int(
            round(plate_height * 1.6))
    else:
        center_of_text_area_y = int(round(plate_center_y)) - int(
            round(plate_height * 1.6))
    text_size_width, text_size_height = text_size

    lower_left_text_origin_x = int(center_of_text_area_x - (text_size_width / 2))
    lower_left_text_origin_y = int(center_of_text_area_y + (text_size_height / 2))

    cv2.putText(img_original_scene, lic_plate.strChars, (lower_left_text_origin_x, lower_left_text_origin_y), font_face,
                font_scale, SCALAR_YELLOW, font_thickness)
# end function

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
# end function

if __name__ == "__main__":
    main()
