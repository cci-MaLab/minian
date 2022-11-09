import os
from natsort import natsorted
import re
import skvideo.io
import numpy as np
import shutil
import csv
from typing import Callable, List, Optional, Union

def fill_video(
    vpath: str,
    pattern=r"msCam[0-9]+\.avi$",
    **kwargs
):
    """
    Fill in missing data across all videos.

    This function first checks whether the videos have been already interpolated by
    checking whether the interpolation_results.csv exists and if the first row of
    that file matches vpath. It will lineraly interpolate between missing frames.
    Therefore indices indicates the last and first good frame. The files will be
    overwritten and their backup will be stored by appending "_orig". Frames missing
    at the beginning or end will simply repeat the first or last known frame respectively.

    We need to load in videos twice for memory purposes. Once to identify which videos
    need fixing then a second time to load all necessary videos into one frame.

    Parameters
    ----------
    vpath : str
        The path containing the videos to load.
    pattern : regexp, optional
        The regexp matching the filenames of the videso. By default
        `r"msCam[0-9]+\.avi$"`, which can be interpreted as filenames starting
        with "msCam" followed by at least a number, and then followed by ".avi".
    """
    # First check if we have already interpolated
    if os.path.isfile(os.path.join(vpath, "interpolation_results.csv")):
        with open(os.path.join(vpath, "interpolation_results.csv"), newline='\n') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == vpath:
                    print("Files already interpolated. Skipping.")
                    return
                else:
                    break

    vpath = os.path.normpath(vpath)
    vlist = natsorted(
        [vpath + os.sep + v for v in os.listdir(vpath) if re.search(pattern, v)]
    )
    if not vlist:
        raise FileNotFoundError(
            "No data with pattern {}"
            " found in the specified folder {}".format(pattern, vpath)
        )
    print("loading {} videos in folder {}".format(len(vlist), vpath))

    # Load in all movies into one dataframe.
    videodata = []
    video_ranges = []
    last_index = 0
    bad_videos = set()
    for i in range(len(vlist)):
        # RBG values are the same so we collapse the final dimension
        video = skvideo.io.vread(vlist[i])[:, :, :, 0]
        if i == 0:
            video[800:1000] = 0
        
        for idx in check_video(video, i, len(vlist) - 1):
            bad_videos.add(idx)
        # Format: first index, last_index
        video_ranges.append([last_index, last_index + len(video) - 1])
        last_index += len(video)

    bad_videos = sorted(bad_videos)
    # Load all the videos to fix in one numpy array
    for i in bad_videos:
        video = skvideo.io.vread(vlist[i])[:, :, :, 0]
        if i == 0:
            video[800:1000] = 0
        videodata.append(video)
    
    videodata = np.vstack(videodata)

    # Check if there is missing data
    indices = get_indices(videodata)

    if indices:
        for start, end in indices:
            # We have several possible edge cases to consider

            # The missing frames are within the video
            if start != 0 and end != len(videodata):
                interpolate(videodata, start, end)
            
            # We are missing frames at the beginning
            if start == -1:
                fill_linear(videodata, start, end)

            # We are missing frames at the end
            if end == len(videodata):
                fill_linear(videodata, start, end)
        
        # Now iterate through all the indices we've affected and overwrite the videos whilst making backups
        i = 0
        j = 0
        overwritten_videos = []
        while i < len(bad_videos) and j < len(indices):
            # Check if ranges overlap
            if video_ranges[bad_videos[i]][0] <= indices[j][0] + 1 <= video_ranges[bad_videos[i]][1] or video_ranges[bad_videos[i]][0] <= indices[j][1] - 1 <= video_ranges[bad_videos[i]][1]:
                video_idx = bad_videos[i]
                overwritten_videos.append(vlist[video_idx])
                name, ext = os.path.basename(vlist[video_idx]).split(".")
                shutil.copyfile(vlist[video_idx], os.path.join(os.path.dirname(vlist[video_idx]), name + "_orig." + ext))
                skvideo.io.vwrite(vlist[video_idx], videodata[video_ranges[bad_videos[i]][0]:video_ranges[bad_videos[i]][1]])
                i += 1
            else:
                j += 1
        
    # Save results in text file
    f = open(os.path.join(vpath, "interpolation_results.csv"), 'w')
    writer = csv.writer(f)
    writer.writerow([vpath])
    if indices:
        writer.writerow(indices)
        writer.writerow(overwritten_videos)
    f.close()

def check_video(video, idx, total_length):
    indices_to_return = []
    frame_max = np.array([np.max(frame) for frame in video])

    if frame_max[0] <= 1:
        if idx != 0:
            indices_to_return.append(idx - 1)
        indices_to_return.append(idx)

    if frame_max[-1] <= 0:
        if idx != total_length:
            indices_to_return.append(idx + 1)
        indices_to_return.append(idx)
    
    if not indices_to_return:
        if (frame_max <= 1).any():
            indices_to_return.append(idx)
    
    return indices_to_return

        

def get_indices(videodata):
    frame_max = np.array([np.max(frame) for frame in videodata])
    # Identify ranges we wish to interpolate
    ranges = []
    indices = frame_max > 1
    valid_indices = np.arange(len(indices))[indices]
    # Prepend -1 if the first value of indices is True
    if not indices[0]:
        valid_indices = np.insert(valid_indices, 0, -1)

    for i in range(len(valid_indices)-1):
        if valid_indices[i] + 1 != valid_indices[i+1]:
            ranges.append([valid_indices[i], valid_indices[i+1]])
    
    if valid_indices[-1] != len(indices) - 1:
        ranges.append([valid_indices[-1], len(indices)])

    return ranges  

def interpolate(videodata, start, end):
    diff = videodata[start].astype("float") - videodata[end].astype("float")
    frames = end - 1 - start
    coefficients = (np.arange(0, frames) + 1) / frames
    interpolated_frames = np.empty((videodata[0:len(coefficients)].shape))
    diff_frames = np.empty((videodata[0:len(coefficients)].shape))
    diff_frames[:] = diff
    interpolated_frames[:] = videodata[start].astype("float")
    
    interpolated_frames -= diff_frames.astype("float") * coefficients[:, None, None]

    videodata[start + 1:end] = interpolated_frames.astype("uint8")

def fill_linear(videodata, start, end):
    if start == -1:
        mirrored_frame = videodata[end]
        start = 0
    else:
        mirrored_frame = videodata[start]

    videodata[start: end] = mirrored_frame