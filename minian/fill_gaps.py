import os
from natsort import natsorted
import re
import skvideo.io
import numpy as np
import shutil
import csv
from subprocess import PIPE, run
import sys

def fill_video(
    vpath: str,
    thresh=20,
    fix_brightness = False,
    pattern=r"[0-9]+\.avi$",
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
    orig_video_ranges = []
    last_index = 0
    bad_videos = set()
    for i in range(len(vlist)):
        # RBG values are the same so we collapse the final dimension
        video = skvideo.io.vread(vlist[i])[:, :, :, 0]
        
        for idx in check_video(video, i, len(vlist) - 1, thresh):
            bad_videos.add(idx)
        # Format: first index, last_index
        orig_video_ranges.append([last_index, last_index + len(video) - 1])
        last_index += len(video)

    bad_videos = sorted(bad_videos)
    video_ranges = []
    last_index = 0
    # Load all the videos to fix in one numpy array
    for i in bad_videos:
        video = skvideo.io.vread(vlist[i])[:, :, :, 0]
        video_ranges.append([last_index, last_index + len(video) - 1])
        last_index += len(video)
        videodata.append(video)
    
    indices = []

    if videodata:
        videodata = np.vstack(videodata)

        # Check if there is missing data
        indices = get_indices(videodata, thresh)

    if indices:
        for start, end in indices:
            # We have several possible edge cases to consider

            # For the glitching effect we need to extend the window a bit
            if np.mean(videodata[start:end, 272:332, 272:332]) > 10:
                start = max(0, start-5)
                end = min(end + 5, len(videodata))
            
            # There is a possibility of a followup glitch with the brigthness increases. This needs to be corrected
            elif fix_brightness and np.mean(videodata[end, 272:332, 272:332]) > 15 + np.mean(videodata[start, 272:332, 272:332]):
                i = end
                last = np.mean(videodata[i, 272:332, 272:332])
                next = np.mean(videodata[i + 1, 272:332, 272:332])
                while next + 15 > last:
                    i += 1
                    last = next
                    next = np.mean(videodata[i + 1, 272:332, 272:332])
                
                # Lower the birghtened section to match the subsequent signal
                diff = last - next
                i += 1
                # The subtraction could cause an overflow, therefore clip the data
                sub_array = np.clip(videodata[end:i].astype("float") - diff, 0, 255)
                videodata[end:i] = sub_array.astype("uint8")

                


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
        adjusted_ranges = []
        while i < len(bad_videos) and j < len(indices):
            # Check if ranges overlap
            if video_ranges[i][0] <= indices[j][0] + 1 <= video_ranges[i][1] or video_ranges[i][0] <= indices[j][1] - 1 <= video_ranges[i][1]:
                video_idx = bad_videos[i]
                overwritten_videos.append(vlist[video_idx])
                name, ext = os.path.basename(vlist[video_idx]).split(".")
                
                try:
                    shutil.copyfile(vlist[video_idx], os.path.join(os.path.dirname(vlist[video_idx]), name + "_orig." + ext))
                except:
                    print("File copy failed for %s" % (vlist[video_idx]))
                    return
                skvideo.io.vwrite(vlist[video_idx], videodata[video_ranges[i][0]:video_ranges[i][1]])
                
                # Fix indices to match global indices
                global_ranges = orig_video_ranges[video_idx]
                local_ranges = video_ranges[i]
                # Only check if the beginning is in the video range
                if video_ranges[i][0] <= indices[j][0] + 1 <= video_ranges[i][1]:
                    diff = global_ranges[0] - local_ranges[0]
                    adjusted_ranges.append([indices[j][0] + diff, indices[j][1] + diff])
                i += 1
            else:
                j += 1
        
    # Save results in text file
    f = open(os.path.join(vpath, "interpolation_results.csv"), 'w')
    writer = csv.writer(f)
    writer.writerow([vpath])
    if indices:        
        writer.writerow(adjusted_ranges)
        writer.writerow(overwritten_videos)
    f.close()

def check_video(video, idx, total_length, thresh):
    indices_to_return = []
    sub_frames = video[:, 272:332, 272:332]
    frame_min = np.array([np.min(frame) for frame in sub_frames])
    # if the value drops x points below median then it's a bad frame
    threshold = np.median(frame_min) - thresh

    if frame_min[0] <= threshold:
        if idx != 0:
            indices_to_return.append(idx - 1)
        indices_to_return.append(idx)

    if frame_min[-1] <= threshold:
        if idx != total_length:
            indices_to_return.append(idx + 1)
        indices_to_return.append(idx)
    
    if not indices_to_return:
        if (frame_min <= threshold).any():
            indices_to_return.append(idx)
    
    return indices_to_return

        

def get_indices(videodata, thresh):
    sub_frames = videodata[:, 272:332, 272:332]
    frame_min = np.array([np.min(frame) for frame in sub_frames])
    # if the value drops 20 points below median then it's a bad frame
    threshold = np.median(frame_min) - thresh
    # Identify ranges we wish to interpolate
    ranges = []
    indices = frame_min > threshold
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
    coefficients = (np.arange(0, frames) + 1) / (frames + 1)
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

def bypass_notebook(vpath: str,
    thresh=20,
    fix_brightness = False,
    pattern=r"[0-9]+\.avi$",
    **kwargs):
    '''
    This function takes in the same parameters as fill_video().
    Unfortunately due to a broken pipe error when running the function in 
    jupyter notebook, the function will need to be invoked through a command line
    call.
    '''
    command = ["python", os.path.abspath(__file__), os.path.abspath(vpath), str(thresh), str(fix_brightness), pattern]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)


if __name__ == "__main__":
    vpath, thresh, fix_brightness, pattern = sys.argv[1:]
    thresh = int(thresh)
    fix_brightness = True if fix_brightness == "True" else False
    fill_video(vpath, thresh, fix_brightness, pattern)