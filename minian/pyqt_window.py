from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QStyle, QWidget, QMessageBox,
                             QPushButton, QLabel, QHBoxLayout, QLineEdit)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIntValidator
from functools import partial
import time
import pyqtgraph as pg
from pyqtgraph import PlotItem, InfiniteLine
import xarray as xr

import numpy as np
from minian.utilities import save_minian

class VideoWindow(QMainWindow):

    def __init__(
            self,
            varr,
            intpath,
            summary=["mean"],
            parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("Video Signal Fix") 

        self.label_video_orig = QLabel()
        self.label_video_updated = QLabel()
        self.label_index = QLabel()
        self.label_index.setFixedWidth(40)
        self.varr_orig = varr
        self.intpath = intpath

        # Video stuff
        self.imv = pg.ImageView()
        self.video = varr.copy()
        self.current_frame = 0
        self.video_length = self.video.shape[0]
        self.imv.setImage(self.video.sel(frame=self.current_frame).values.T)

        # Summary
        self._summary_types = summary.copy()

        # Visualize signals selected in video
        self.w_signals = pg.GraphicsLayoutWidget()
        self.w_signals.setFixedHeight(200)
        self.pi = PlotItem()
        self.pi.getViewBox().setMouseEnabled(x=False, y=False)
        self.pi.setYRange(0, 255, padding=0)
        self.w_signals.addItem(self.pi, row=0, col=0)
        self.pi.addLegend(offset=(-30, 30))
        self.color_options = ["r", "g", "b", "y", "c", "m", "w"]
        self.plotLine = InfiniteLine(pos=0, angle=90, movable=True, bounds=[0, self.video_length-1])
        self.plotLine.sigDragged.connect(self.dragging)
        self.pi.setTitle(f"Signals for {self._summary_types}")
        self.reset_plot()         

        # Play Button
        self.playButton = QPushButton()
        self.playButton.setCheckable(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play_pause)

        # Forward and Backward Buttons
        self.forwardButton = QPushButton()
        self.forwardButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        self.forwardButton.clicked.connect(self.clicked_next)

        self.backwardButton = QPushButton()
        self.backwardButton.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.backwardButton.clicked.connect(self.clicked_prev)

        # Frame fixing elements
        self.start_label_fix = QLabel()
        self.start_label_fix.setText("Select Start Fix:")
        self.startbox_fix = QLineEdit(self)
        self.end_label_fix = QLabel()
        self.end_label_fix.setText("Select End Fix:")
        self.endbox_fix = QLineEdit(self)
        self.interpolate_button = QPushButton()
        self.interpolate_button.setText("Interpolate")
        self.interpolate_button.clicked.connect(self.interpolate)
        self.fix_brightness_button = QPushButton()
        self.fix_brightness_button.setText("Fix brightness")
        #self.fix_brightness_button.clicked.connect()
        self.reset_button = QPushButton()
        self.reset_button.setText("Reset")
        self.reset_button.clicked.connect(self.reset)
        self.linear_left_button = QPushButton()
        self.linear_left_button.setText("Linear Left")
        self.linear_left_button.clicked.connect(self.fill_left)
        self.linear_right_button = QPushButton()
        self.linear_right_button.setText("Linear Right")
        self.linear_right_button.clicked.connect(self.fill_right)

        # Saving 
        self.save_button = QPushButton()
        self.save_button.setText("Save Changes")
        self.save_button.clicked.connect(self.save)
        self.save_label = QLabel()
        self.save_label.setText("")

        
        # Set up Layout
        wid = QWidget(self)
        self.setCentralWidget(wid)

        videoLayout = QVBoxLayout()
        videoLayout.addWidget(self.imv)
        videoLayout.addWidget(self.w_signals)

        controlLayout  = QHBoxLayout()
        controlLayout.addWidget(self.backwardButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.forwardButton)
        controlLayout.addWidget(self.label_index)

        fixLayout = QHBoxLayout()
        fixLayout.addWidget(self.start_label_fix)
        fixLayout.addWidget(self.startbox_fix)
        fixLayout.addWidget(self.end_label_fix)
        fixLayout.addWidget(self.endbox_fix)
        fixLayout.addWidget(self.interpolate_button)
        fixLayout.addWidget(self.fix_brightness_button)
        fixLayout.addWidget(self.linear_left_button)
        fixLayout.addWidget(self.linear_right_button)
        fixLayout.addWidget(self.reset_button)

        saveLayout = QHBoxLayout()
        saveLayout.addWidget(self.save_button)
        saveLayout.addWidget(self.save_label)



        centralLayout = QVBoxLayout()
        centralLayout.addLayout(videoLayout)
        centralLayout.addLayout(controlLayout)
        centralLayout.addLayout(fixLayout)
        centralLayout.addLayout(saveLayout)

        layout = QHBoxLayout()
        layout.addLayout(centralLayout)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.video_timer = QTimer()
        self.video_timer.setInterval(50)
        self.video_timer.timeout.connect(self.next_frame)

        

    def clicked_next(self):
        self.pause_video()
        self.next_frame()
    
    def clicked_prev(self):
        self.pause_video()
        self.prev_frame()

    def play_pause(self):
        if self.playButton.isChecked():
            self.start_video()
        else:
            self.pause_video()

    def dragging(self):
        self.pause_video()
        self.current_frame = int(self.plotLine.value())
        self.label_index.setText(str(self.current_frame))  
        image = self.video.sel(frame=self.current_frame).values
        self.imv.setImage(image.T, autoRange=False, autoLevels=False)
        self.playButton.setChecked(False)


    def next_frame(self):
        self.current_frame = (1 + self.current_frame) % self.video_length
        self.plotLine.setValue(self.current_frame)
        image = self.video.sel(frame=self.current_frame).values
        self.imv.setImage(image.T, autoRange=False, autoLevels=False)
        self.label_index.setText(str(self.current_frame)) 

    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.video_length
        self.plotLine.setValue(self.current_frame)
        image = self.video.sel(frame=self.current_frame).values
        self.imv.setImage(image.T, autoRange=False, autoLevels=False)
        self.label_index.setText(str(self.current_frame)) 
    
    def pause_video(self):
        self.video_timer.stop()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
    
    def start_video(self):
        self.video_timer.start()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def save(self):
        save_minian(
        self.video.rename("varr_fixed"),
        self.intpath,
        overwrite=True,
    )
        
    def interpolate(self):
        start = int(self.startbox_fix.text()) if self.startbox_fix.text() else -1
        end = int(self.endbox_fix.text()) if self.endbox_fix.text() else -1
        if start >= 0 and end >= 0 and start < end:
            self.pause_video()
            first_frame = self.video.sel(frame=start).values.astype("float")
            last_frame = self.video.sel(frame=end).values.astype("float")

            diff = first_frame - last_frame
            frames = end-start-1
            coefficients = (np.arange(0, frames) + 1) / (frames + 2)
            interpolated_frames = np.empty((len(coefficients), first_frame.shape[0], first_frame.shape[1]))
            diff_frames = np.empty((len(coefficients), first_frame.shape[0], first_frame.shape[1]))
            diff_frames[:] = diff
            interpolated_frames[:] = first_frame           
            interpolated_frames -= diff_frames.astype("float") * coefficients[:, None, None]
            self.video.data[start+1:end] = interpolated_frames.astype("uint8")
        
            image = self.video.sel(frame=self.current_frame).values
            self.imv.setImage(image.T, autoRange=False, autoLevels=False)
            self.reset_plot()
        else:
            self.error_message()

    def fill_right(self):
        start = int(self.startbox_fix.text()) if self.startbox_fix.text() else -1
        end = int(self.endbox_fix.text()) if self.endbox_fix.text() else -1
        if start >= 0 and end >= 0 and start < end:
            self.pause_video()
            self.video.data[start+1:end] = self.video.sel(frame=start).values
            image = self.video.sel(frame=self.current_frame).values
            self.imv.setImage(image.T, autoRange=False, autoLevels=False)
            self.reset_plot()
        else:
            self.error_message()



    def fill_left(self):
        start = int(self.startbox_fix.text()) if self.startbox_fix.text() else -1
        end = int(self.endbox_fix.text()) if self.endbox_fix.text() else -1
        if start >= 0 and end >= 0 and start < end:
            self.pause_video()
            self.video.data[start:end-1] = self.video.sel(frame=end).values
            image = self.video.sel(frame=self.current_frame).values
            self.imv.setImage(image.T, autoRange=False, autoLevels=False)
            self.reset_plot()
        else:
            self.error_message()

    def error_message(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Selected ranges are invalid.")
        msg.setInformativeText('Make sure the selected ranges are sensible.')
        msg.setWindowTitle("Error")
        msg.exec_()

    def reset(self):
        self.pause_video()
        self.video = self.varr_orig.copy()
        self.reset_plot()
        image = self.video.sel(frame=self.current_frame).values
        self.imv.setImage(image.T, autoRange=False, autoLevels=False)

    def reset_plot(self):
        self.pi.clear()

        self.summ_all = {
            "mean": self.video.mean(["height", "width"]),
            "max": self.video.max(["height", "width"]),
            "min": self.video.min(["height", "width"]),
            "diff": self.video.diff("frame").mean(["height", "width"]),
            "residual": xr.apply_ufunc(np.fabs,
                                    (self.video.astype(np.int16) - self.video.sel(frame=0).astype(np.int16))*10,
                                    dask="parallelized")
                                    .mean(["height", "width"])
        }
        try:
            self.summ = {k: self.summ_all[k] for k in self._summary_types}
        except KeyError:
            print("{} Not understood for specifying summary".format(self._summary_types))
        if self.summ:
            print("computing summary")
            sum_list = []
            for k, v in self.summ.items():
                sum_list.append(v.compute().assign_coords(sum_var=k))
            summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        for i, summ in enumerate(self._summary_types):
            summary = self.summary.sel(sum_var=summ).values
            # Plot the summary with a specific color
            self.pi.plot(summary, name=summ, pen=pg.mkColor(self.color_options[i%len(self.color_options)]))  
        self.pi.addItem(self.plotLine)