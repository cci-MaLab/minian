from inspect import currentframe
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QStyle, QSizePolicy, QWidget, QMessageBox,
                             QPushButton, QSlider, QLabel, QApplication, QHBoxLayout, QLineEdit)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtGui import QImage, QPixmap
from functools import partial
import time




from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure

import numpy as np
import skvideo.io
import ffmpeg
import shutil
import os

import sys
from minian.frame_fix_utility import *

class MplCanvas(Canvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

class MplWidget(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvas(width=width, height=height, dpi=dpi)                  # Create canvas object
        self.vbl = QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

class ThreadDisplay(QThread):    
    changePixmap = pyqtSignal(QImage)
    def __init__(self):
        super().__init__()
        self.frame = 0
        self.last_frame = -1
        

    def set_frame(self, frame):
        self.frame = frame
    
    def set_video(self, video):
        self.video = video

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if self.frame != self.last_frame:
                h, w = self.video[self.frame].shape
                
                image = QImage(self.video[self.frame], w, h, w, QImage.Format_Grayscale8)
                image = image.scaled(480, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(image)
                self.last_frame = self.frame
                
class ThreadFrameSetter(QThread):
    frame = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.current_frame = 0
        self.limit = 1000
        self.playing = False

    def set_frame(self, frame):
        self.current_frame = frame

    def set_limit(self, limit):
        self.limit = limit
    
    def stop(self):
        self.playing = False

    def run(self):
        self.playing = True
        last_time = time.time()
        thresh = 1 / 30
        while self.playing:
            diff = time.time() - last_time

            if diff > thresh:
                last_time = time.time()
                if self.current_frame < self.limit - 1:
                    self.current_frame += 1
                    self.frame.emit(self.current_frame)
            

class VideoWindow(QMainWindow):

    def __init__(self, mean, max, orig_video_ranges, vlist, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("Video Signal Fix") 
        self.orig_video_ranges = orig_video_ranges
        self.vlist = vlist

        self.label_video_orig = QLabel()
        self.label_video_updated = QLabel()
        self.label_index = QLabel()
        self.label_index.setFixedWidth(40)
        self.range = 0
        self.videodata = None
        self.videodata_new = None
        self.implemented_changes = {}
        self.loaded_videos = []

        self.video_orig_thread = ThreadDisplay()
        self.video_orig_thread.changePixmap.connect(self.setImageOrig)
        self.video_new_thread = ThreadDisplay()
        self.video_new_thread.changePixmap.connect(self.setImageNew)
        self.frame_setter_thread = ThreadFrameSetter()
        self.frame_setter_thread.frame.connect(self.setFrame)


        # Plotting
        self.mpl_plot = MplWidget(self, width=5, height=4, dpi=100)
        self.mpl_plot.canvas.ax.plot(mean, color='b')
        self.mpl_plot.canvas.ax.plot(max, color='r')
        self.toolbar = NavigationToolbar(self.mpl_plot.canvas, self)

        # Specify Ranges
        self.start_label = QLabel()
        self.start_label.setText("Select Start Range:")
        self.startbox = QLineEdit(self)
        self.startbox.setValidator(QIntValidator(0, len(mean), self))
        self.end_label = QLabel()
        self.end_label.setText("Select End Range:")
        self.endbox = QLineEdit(self)
        self.endbox.setValidator(QIntValidator(0, len(mean), self))
        self.range_button = QPushButton()
        self.range_button.setEnabled(True)
        self.range_button.setText("Load video")
        self.range_button.clicked.connect(self.load_video)

        # Play Button
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        # Forward and Backward Buttons
        self.forwardButton = QPushButton()
        self.forwardButton.setEnabled(False)
        self.forwardButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.forwardButton.clicked.connect(self.step_forward)

        self.backwardButton = QPushButton()
        self.backwardButton.setEnabled(False)
        self.backwardButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.backwardButton.clicked.connect(self.step_backward)

        # Slider
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # Frame fixing elements
        self.start_label_fix = QLabel()
        self.start_label_fix.setText("Select Start Fix:")
        self.startbox_fix = QLineEdit(self)
        self.startbox_fix.setValidator(QIntValidator(0, len(mean), self))
        self.startbox_fix.setEnabled(False)
        self.end_label_fix = QLabel()
        self.end_label_fix.setText("Select End Fix:")
        self.endbox_fix = QLineEdit(self)
        self.endbox_fix.setValidator(QIntValidator(0, len(mean), self))
        self.endbox_fix.setEnabled(False)
        self.interpolate_button = QPushButton()
        self.interpolate_button.setEnabled(False)
        self.interpolate_button.setText("Interpolate")
        self.interpolate_button.clicked.connect(partial(self.update_frame, "interpolate"))
        self.fix_brightness_button = QPushButton()
        self.fix_brightness_button.setEnabled(False)
        self.fix_brightness_button.setText("Fix brightness")
        self.fix_brightness_button.clicked.connect(partial(self.update_frame, "fix_brightness"))
        self.revert_button = QPushButton()
        self.revert_button.setEnabled(False)
        self.revert_button.setText("Revert")
        self.revert_button.clicked.connect(self.revert)
        self.linear_left_button = QPushButton()
        self.linear_left_button.setEnabled(False)
        self.linear_left_button.setText("Linear Left")
        self.linear_left_button.clicked.connect(partial(self.update_frame, "linear_left"))
        self.linear_right_button = QPushButton()
        self.linear_right_button.setEnabled(False)
        self.linear_right_button.setText("Linear Right")
        self.linear_right_button.clicked.connect(partial(self.update_frame, "linear_right"))

        # Saving 
        self.save_button = QPushButton()
        self.save_button.setEnabled(False)
        self.save_button.setText("Save Changes")
        self.save_button.clicked.connect(self.save)
        self.save_label = QLabel()
        self.save_label.setText("")

        
        # Set up Layout
        wid = QWidget(self)
        self.setCentralWidget(wid)

        rangeLayout = QHBoxLayout()
        rangeLayout.addWidget(self.start_label)
        rangeLayout.addWidget(self.startbox)
        rangeLayout.addWidget(self.end_label)
        rangeLayout.addWidget(self.endbox)
        rangeLayout.addWidget(self.range_button)

        videoLayout = QHBoxLayout()
        videoLayout.addWidget(self.label_video_orig)
        videoLayout.addWidget(self.label_video_updated)

        controlLayout  = QHBoxLayout()
        controlLayout.addWidget(self.backwardButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.forwardButton)
        controlLayout.addWidget(self.label_index)
        controlLayout.addWidget(self.positionSlider)

        fixLayout = QHBoxLayout()
        fixLayout.addWidget(self.start_label_fix)
        fixLayout.addWidget(self.startbox_fix)
        fixLayout.addWidget(self.end_label_fix)
        fixLayout.addWidget(self.endbox_fix)
        fixLayout.addWidget(self.interpolate_button)
        fixLayout.addWidget(self.fix_brightness_button)
        fixLayout.addWidget(self.linear_left_button)
        fixLayout.addWidget(self.linear_right_button)
        fixLayout.addWidget(self.revert_button)

        saveLayout = QHBoxLayout()
        saveLayout.addWidget(self.save_button)
        saveLayout.addWidget(self.save_label)



        centralLayout = QVBoxLayout()
        centralLayout.addWidget(self.toolbar)
        centralLayout.addWidget(self.mpl_plot)
        centralLayout.addLayout(rangeLayout)
        centralLayout.addLayout(videoLayout)
        centralLayout.addLayout(controlLayout)
        centralLayout.addLayout(fixLayout)
        centralLayout.addLayout(saveLayout)

        layout = QHBoxLayout()
        layout.addLayout(centralLayout)

        # Set widget to contain window contents
        wid.setLayout(layout)


        
    def load_video(self):
        start, end = int(self.startbox.text()), int(self.endbox.text())
        if start >= end or end - start > 3000:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Selected ranges are invalid.")
            msg.setInformativeText('Make sure start is less than end and that the distance is less than 3000 frames.')
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            self.video_orig_thread.stop()
            self.video_new_thread.stop()
            self.frame_setter_thread.stop()
            self.frame_setter_thread.set_frame(0)
            self.positionSlider.setValue(0)
            # Find the correct videos
            self.start_idx = -1
            first_vid = -1
            last_vid = -1
            for i in range(len(self.orig_video_ranges)):
                if self.orig_video_ranges[i][0] <= start <= self.orig_video_ranges[i][1]:
                    first_vid = i
                    self.start_idx = self.orig_video_ranges[i][0] 
                    break
            
            while i < len(self.orig_video_ranges):
                if self.orig_video_ranges[i][0] <= end <= self.orig_video_ranges[i][1]:
                    last_vid = i
                    break
                i += 1
            
            # Make note of loaded videos
            for j in range(first_vid, last_vid + 1):
                self.loaded_videos.append(j)

            # Load the videos into numpy dataframe
            self.videodata = []
            self.framerate = 20
            for i in range(first_vid, last_vid + 1):
                video = skvideo.io.vread(self.vlist[i])[:, :, :, 0]
                self.framerate = skvideo.io.ffprobe(self.vlist[i])['video']['@r_frame_rate']
                self.videodata.append(video)

            self.videodata = np.vstack(self.videodata)
            self.range = len(self.videodata) - 1
            self.positionSlider.setRange(0, self.range)
            self.label_index.setText("%i" % (self.start_idx))
            self.startbox_fix.clear()
            self.endbox_fix.clear()
            self.startbox_fix.setEnabled(True)
            self.endbox_fix.setEnabled(True)
            self.startbox_fix.setValidator(QIntValidator(self.start_idx, self.range + self.start_idx, self))
            self.endbox_fix.setValidator(QIntValidator(self.start_idx, self.range + self.start_idx, self))
            

            
            # Now check our list to see if we need to update it
            for key, value in self.implemented_changes.items():
                start, end = key.split(',')
                start, end = int(start) - self.start_idx, int(end) - self.start_idx

                x = range(start, end)
                y = range(0, self.range)

                overlap = list(set(x) & set(y))
                overlap.sort()

                if overlap:
                    # Check the indices of the overlap
                    new_indices = np.searchsorted(range(start, end), overlap)
                    self.videodata[overlap] = value[0][new_indices]

            self.videodata_new = self.videodata.copy()

            # Store it locally for now to use later
            self.video_orig_thread.set_video(self.videodata)
            self.video_orig_thread.start()


            self.playButton.setEnabled(True)
            self.forwardButton.setEnabled(True)
            self.backwardButton.setEnabled(True)
            self.linear_left_button.setEnabled(True)
            self.linear_right_button.setEnabled(True)
            self.interpolate_button.setEnabled(True)
            self.fix_brightness_button.setEnabled(True)
            self.revert_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.frame_setter_thread.set_limit(self.range)

            
            
    def update_frame(self, term):
        # First check if the values are sensible
        start, end = int(self.startbox_fix.text()), int(self.endbox_fix.text())
        if (start < end) and (start >= self.start_idx) and (end < self.start_idx + self.range):            
            videodata_orig = self.videodata_new[start:end+1].copy()

            if term == "interpolate":
                updated_segment = interpolate(self.videodata_new[start:end+1])
            elif term == "fix_brightness":
                updated_segment = fix_brightness(self.videodata_new[start:end+1])
            elif term == "linear_left":
                updated_segment = fill_linear(self.videodata_new[start:end+1], left=True)
            elif term == "linear_right":
                updated_segment = fill_linear(self.videodata_new[start:end+1], left=False)
            
            #self.videodata_new[start:end+1] = updated_segment
            self.video_new_thread.set_video(self.videodata_new)
            self.video_new_thread.start()

            # Store the changes
            self.implemented_changes[str(start + self.start_idx) + "," + str(end + self.start_idx)] = (updated_segment, videodata_orig)

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Selected ranges are invalid.")
            msg.setInformativeText('Make sure the selected ranges are sensible.')
            msg.setWindowTitle("Error")
            msg.exec_()

    def revert(self):
        if self.implemented_changes:
            key, value = list(self.implemented_changes.items())[-1]
            start, end = key.split(',')
            start, end = int(start) - self.start_idx, int(end) - self.start_idx

            x = range(start, end)
            y = range(0, self.range)

            overlap = list(set(x) & set(y))
            overlap.sort()
            if overlap:
                new_indices = np.searchsorted(range(start, end), overlap)
                self.videodata_new[overlap] = value[1][new_indices]
                self.implemented_changes.popitem()
            
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Can't revert last change")
                msg.setInformativeText('The last change is outside of the current observed range.')
                msg.setWindowTitle("Error")
                msg.exec_()
            
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Can't revert last change")
            msg.setInformativeText('No changes have been done.')
            msg.setWindowTitle("Error")
            msg.exec_()



    def play(self):
        if self.frame_setter_thread.playing:
            self.frame_setter_thread.stop()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.frame_setter_thread.start()
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))

    def save(self):
        if self.implemented_changes:
            # We need to kill the threads for this to work
            self.video_orig_thread.stop()
            self.video_new_thread.stop()
            self.frame_setter_thread.stop()
            self.frame_setter_thread.set_frame(0)
            self.positionSlider.setValue(0)

            
            self.save_label.setText("Saving")
            video_list = set(self.loaded_videos)
            for i in video_list:
                video = skvideo.io.vread(self.vlist[i])[:, :, :, 0]
                # Check for potential changes
                to_delete = []
                for key, value in self.implemented_changes.items():
                    start_idx = self.orig_video_ranges[i][0]
                    start, end = key.split(',')
                    start, end = int(start) - start_idx, int(end) - start_idx

                    x = range(start, end)
                    y = range(0, self.orig_video_ranges[i][1] - self.orig_video_ranges[i][0])

                    overlap = list(set(x) & set(y))
                    overlap.sort()

                    if overlap:
                        # Check the indices of the overlap
                        new_indices = np.searchsorted(range(start, end), overlap)
                        video[overlap] = value[0][new_indices]
                        to_delete.append(key)
                
                for key in to_delete:
                    del self.implemented_changes[key]
                
                name, ext = os.path.basename(self.vlist[i]).split(".")
                try:
                    shutil.copyfile(self.vlist[i], os.path.join(os.path.dirname(self.vlist[i]), name + "_orig." + ext))
                    write_vid(self.vlist[i], video, self.framerate)
                except:
                    print("File copy failed for %s" % (self.vlist[i]))
                    return

            self.save_label.setText("Saved")
            self.video_orig_thread.start()
            self.video_new_thread.start()



    def step_forward(self):
        self.frame_setter_thread.stop()
        self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
        frame = np.clip(self.positionSlider.value() + 1, 0, self.range)
        self.positionSlider.setValue(frame)
        self.video_orig_thread.set_frame(frame)
        self.video_new_thread.set_frame(frame)
        self.frame_setter_thread.set_frame(frame)
        self.label_index.setText("%i" % (self.start_idx + frame))

    def step_backward(self):
        self.frame_setter_thread.stop()
        self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
        frame = np.clip(self.positionSlider.value() - 1, 0, self.range)
        self.positionSlider.setValue(frame)
        self.video_orig_thread.set_frame(frame)
        self.video_new_thread.set_frame(frame)
        self.frame_setter_thread.set_frame(frame)
        self.label_index.setText("%i" % (self.start_idx + frame))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        self.label_index.setText("%i" % (self.start_idx + position))

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
        self.duration = duration
    
    def setPosition(self, position):
        self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))
        self.frame_setter_thread.stop()
        self.video_orig_thread.set_frame(position)
        self.video_new_thread.set_frame(position)
        self.frame_setter_thread.set_frame(position)
        self.label_index.setText("%i" % (self.start_idx + position))

    def setImageOrig(self, image):
        self.label_video_orig.setPixmap(QPixmap.fromImage(image))
    
    def setImageNew(self, image):
        self.label_video_updated.setPixmap(QPixmap.fromImage(image))

    def setFrame(self, frame):
        self.positionSlider.setValue(frame)
        self.video_orig_thread.set_frame(frame)
        self.video_new_thread.set_frame(frame)
        self.label_index.setText("%i" % (self.start_idx + frame))

def write_vid(vpath, arr, framerate, options={"crf": "18", "preset": "ultrafast"}):
    """
    This is a simplified and refactored version from write_video function
    in visualization.py
    """
    vpath = vpath[:-3] + 'mkv'
    w, h = arr.shape[1:]
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s="{}x{}".format(w, h))
        .output(vpath, pix_fmt="yuv420p", vcodec="libx264", r=framerate, **options)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    process.stdin.write(arr.tobytes())
    process.stdin.close()
    process.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoWindow()
    player.resize(1500, 745)
    player.show()
    sys.exit(app.exec_())