# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLineEdit, QLabel
from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
import sys
import cv2
import numpy as np
import time
import settings

poseEstimator = None

 
def load_model():
    global poseEstimator
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_thin'), target_size=(432, 368))


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        self.__layout_time_input_from = QtWidgets.QHBoxLayout()
        self.__layout_time_input_to = QtWidgets.QHBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'Realtime (Camera) OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'Attitude estimation OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'Multiplayer tracking OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'Behavior recognition OFF')
        self.textbox_from = QLineEdit(self)
        self.textbox_to = QLineEdit(self)
        self.nameLabel_from = QLabel(self)
        self.nameLabel_to = QLabel(self)
       # self.nameLabel_uploadInfo = QLabel(self)
        
        
        self.button_mode_4 = QtWidgets.QPushButton(u'Upload prerecorded file')
        self.button_mode_5 = QtWidgets.QPushButton(u'Face recognition OFF')

        self.button_close = QtWidgets.QPushButton(u'Exit')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)
        self.button_mode_4.setMinimumHeight(50)
        self.button_mode_5.setMinimumHeight(50)
        self.textbox_from.setMinimumHeight(35)
        self.textbox_to.setMinimumHeight(35)
        self.nameLabel_from.setText('FROM:  ')
        self.nameLabel_to.setText('To:       ')
      #  self.nameLabel_uploadInfo.setText('For Scanning a prerecorded footage ( insert time period (from and to) to narrow down search):')
       

        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)

        # self.infoBox = QtWidgets.QTextBrowser(self)
        # self.infoBox.setGeometry(QtCore.QRect(10, 300, 200, 180))

     
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200,200)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)
	

        self.__layout_time_input_from.addWidget(self.nameLabel_from)
        self.__layout_time_input_from.addWidget(self.textbox_from)
        self.__layout_time_input_to.addWidget(self.nameLabel_to)
        self.__layout_time_input_to.addWidget(self.textbox_to)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_mode_5)
        #self.__layout_fun_button.addWidget(self.nameLabel_uploadInfo)
        self.__layout_fun_button.addLayout(self.__layout_time_input_from)
        self.__layout_fun_button.addLayout(self.__layout_time_input_to)
        
        self.__layout_fun_button.addWidget(self.button_mode_4)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'Anomaly Detection using Action Recognition')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_mode_5.clicked.connect(self.button_event)
        #self.button_mode_4.clicked.connect(self.button_event)   Upload button
        self.button_close.clicked.connect(self.close)

    def button_event(self):
        sender = self.sender()
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'Attitude estimation ON')
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.button_mode_3.setText(u'Behavior recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Attitude estimation OFF')
               # self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'Attitude estimation OFF')
                self.button_mode_2.setText(u'Multiplayer tracking ON')
                self.button_mode_3.setText(u'Behavior recognition OFF')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
               # self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_5 and self.timer_camera.isActive():
            if self.__flag_mode != 5:
                self.__flag_mode = 5
                self.button_mode_1.setText(u'Attitude estimation OFF')
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.button_mode_3.setText(u'Behavior recognition OFF')
                self.button_mode_5.setText(u'Face recognition ON')
            else:
                self.__flag_mode = 0
                self.button_mode_5.setText(u'Face recognition OFF')
               # self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'Attitude estimation OFF')
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.button_mode_3.setText(u'Behavior recognition ON')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'Behavior recognition OFF')
               # self.infoBox.setText(u'Camera is on')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'Attitude estimation OFF')
            self.button_mode_2.setText(u'Multiplayer tracking OFF')
            self.button_mode_3.setText(u'Behavior recognition OFF')
            self.button_mode_5.setText(u'Face recognition OFF')
            if self.timer_camera.isActive() == False:
                flag = self.cap.open(self.CAM_NUM)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check if the camera and computer are connected correctly",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(1)
                    self.button_open_camera.setText(u'Realtime (Camera) ON')
                  #  self.infoBox.setText(u'Camera is on')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'Realtime (Camera) OFF')
               # self.infoBox.setText(u'Camera is off')

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        if ret:
            if self.__flag_mode == 1:
              #  self.infoBox.setText(u'Current human body Attitude estimation Mode')
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)

            elif self.__flag_mode == 2:
              #  self.infoBox.setText(u'Currently Multiplayer tracking Mode')
                humans = poseEstimator.inference(show)
                show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)

                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            elif self.__flag_mode == 3:
              #  self.infoBox.setText(u'Current human body Behavior recognition Mode')
                humans = poseEstimator.inference(show)
                ori = np.copy(show)
                show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)
                    self.current = [i[-1] for i in trackers]

                    if len(self.previous) > 0:
                        for item in self.previous:
                            if item not in self.current and item in self.data:
                                del self.data[item]
                            if item not in self.current and item in self.memory:
                                del self.memory[item]

                    self.previous = self.current
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        try:
                            j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        except:
                            j = 0
                        if joint_filter(joints[j]):
                            joints[j] = joint_completion(joint_completion(joints[j]))
                            if label not in self.data:
                                self.data[label] = [joints[j]]
                                self.memory[label] = 0
                            else:
                                self.data[label].append(joints[j])

                            if len(self.data[label]) == settings.L:
                                pred = actionPredictor().move_status(self.data[label])
                                if pred == 0:
                                    pred = self.memory[label]
                                else:
                                    self.memory[label] = pred
                                self.data[label].pop(0)

                                location = self.data[label][-1][1]
                                if location[0] <= 30:
                                    location = (51, location[1])
                                if location[1] <= 10:
                                    location = (location[0], 31)

                                cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            elif self.__flag_mode == 5:
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
                cap = cv2.VideoCapture(0) 
				  
				# loop runs if capturing has been initialized. 
                while 1:  
                    ret, img = cap.read()  
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				  
                    for (x,y,w,h) in faces: 
				        # To draw a rectangle in a face  
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]			        
                        eyes = eye_cascade.detectMultiScale(roi_gray)  
				  
				        #To draw a rectangle in eyes 
                        for (ex,ey,ew,eh) in eyes: 
                            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),2) 
				  
				    # Display an image in a window 
                    cv2.imshow('img',img) 
				  
				    # Wait for Esc key to stop 
                    k = cv2.waitKey(30) & 0xff
                    if k == 27: 
                        break
				  
				# Close the window 
                cap.release() 
				  
				# De-allocate any associated memory usage 
                cv2.destroyAllWindows()  

            end = time.time()
            self.fps = 1. / (end - start)
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"Shut down", u"Confirm Shut down！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Yes')
        cancel.setText(u'No')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


if __name__ == '__main__':
    load_model()
    print("Load all models done!")
    print("The system starts oo run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())