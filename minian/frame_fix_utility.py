import numpy as np

def interpolate(video):
    diff = videodata[0].astype("float") - videodata[-1].astype("float")
    frames = len(videodata)-2
    coefficients = (np.arange(0, frames) + 1) / (frames + 2)
    interpolated_frames = np.empty((videodata[0:len(coefficients)].shape))
    diff_frames = np.empty((videodata[0:len(coefficients)].shape))
    diff_frames[:] = diff
    interpolated_frames[:] = videodata[0].astype("float")
    
    interpolated_frames -= diff_frames.astype("float") * coefficients[:, None, None]

    videodata[1:-1] = interpolated_frames.astype("uint8")

    return videodata.copy()

def fill_right(video, first, last):
    video.data[first+1:last] = self.varr_copy.sel(frame=self._first).values

def fill_left(video, first, last):
    if left:
        videodata[:] = videodata[-1]
    else:
        videodata[:] = videodata[0]

def fix_brightness(videodata):
   
    # Lower the birghtened section to match the subsequent signal
    diff = np.mean(videodata[-2]) - np.mean(videodata[-1]) 
    # The subtraction could cause an overflow, therefore clip the data
    videodata = np.clip(videodata[1:-1].astype("float") - diff, 0, 255)
    
    return videodata.copy()