# plate-recognition.py

import cv2
import numpy as np
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import DetectChars
import DetectPlates
import PossiblePlate
from PIL import Image
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


class MainWindow(QtWidgets.QMainWindow):
    plate_image_file= str()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Choose.clicked.connect(self.showDialog)
        self.Recognition.clicked.connect(self.licensePlateRecognition)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(424, 409)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setMinimumSize(QtCore.QSize(0, 380))
        self.scrollArea.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 404, 378))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout.setObjectName("gridLayout")
        self.Choose = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.Choose.setObjectName("Choose")
        self.gridLayout.addWidget(self.Choose, 3, 0, 1, 1)
        self.Recognition = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.Recognition.setObjectName("Recognition")
        self.gridLayout.addWidget(self.Recognition, 3, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents_2)
        self.verticalLayout.addWidget(self.scrollArea)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Choose.setText(_translate("MainWindow", "Выбрать фото"))
        self.Recognition.setText(_translate("MainWindow", "Распознать"))

    def showDialog(self):
        self.plate_image_file = QtWidgets.QFileDialog.getOpenFileName(self, str("Open Image"), "/home", str("Image Files (*.png *.jpg *.bmp)"))
        file = ''.join(self.plate_image_file[0])
        pixmap=QtGui.QPixmap(file)
        self.label.setPixmap(pixmap)
        self.label.resize(pixmap.width(), pixmap.height())
        self.resize(pixmap.width(), pixmap.height())

    def licensePlateRecognition(self):

        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()  

        if blnKNNTrainingSuccessful == False: 
            print("\nerror: KNN traning was not successful\n")  
            return  

        file=''.join(self.plate_image_file[0])
        imgOriginalScene = cv2.imread(file)  

        if imgOriginalScene is None:  # if image was not read successfully
            print("\nerror: image not read from file \n\n") 
            exit(-1)
            return 

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)  

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  

        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")  
        else:  
           
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            licPlate = listOfPossiblePlates[0]

            if len(licPlate.strChars) == 0: 
                print("\nno characters were detected\n\n")  
                return  

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)  

        print("\nlicense plate read from image = " + licPlate.strChars + "\n") 
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)  
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)  
        pixmap = QtGui.QPixmap("imgOriginalScene.png")
        self.label.setPixmap(pixmap)


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2) 
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0 
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0  
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX  
    fltFontScale = float(plateHeight) / 30.0 
    intFontThickness = int(round(fltFontScale * 1.5)) 

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale,
                                         intFontThickness)  


    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX) 
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX) 

    if intPlateCenterY < (sceneHeight * 0.75): 
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(
            round(plateHeight * 1.6))  
    else:  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(
            round(plateHeight * 1.6)) 
    textSizeWidth, textSizeHeight = textSize  

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))   
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))  

    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace,
                fltFontScale, SCALAR_YELLOW, intFontThickness)



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
