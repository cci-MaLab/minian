from typing import Optional, Union, List
import xarray as xr
import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import cm
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews.util import Dynamic
import panel as pn
from scipy.ndimage import gaussian_filter1d
from holoviews.streams import Stream
import param
from scipy.signal import find_peaks
import os

from minian.utilities import (
    open_minian,
    match_information,
    match_path
)



class CellClustering:
    """
    Cell clustering class. This class is used to cluster cells based on their
    temporal activity, using FFT and agglomerative clustering.
    """

    def __init__(
        self,
        minian: Optional[xr.Dataset] = None,
        signals: Optional[xr.Dataset] = None,
        A: Optional[xr.DataArray] = None,
        fft: bool = True
    ):
    
        self.signals = signals if signals is not None else minian['C']
        self.A = A if A is not None else minian['A']
        self.units = self.signals['unit_id'].values
        self.psd_list = []

        if fft:
            for unit in self.units:
                self.compute_psd(unit) # compute psd for each unit
        else:
            self.psd_list = [self.signals.sel(unit_id=unit).values for unit in self.units]
        
        # Compute agglomerative clustering
        self.linkage_data = linkage(self.psd_list, method='average', metric='cosine')

    def compute_psd(self, unit: int):
        val = self.signals.sel(unit_id=unit).values
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list.append(psd)
    
    def visualize_dendrogram(self, color_threshold=None):
        self.dendro = dendrogram(self.linkage_data, color_threshold=color_threshold)
        return self.dendro
        
    
    def visualize_clusters(self, distance):
        self.cluster_indices = fcluster(self.linkage_data, distance, criterion='distance')
        viridis = cm.get_cmap('viridis', self.cluster_indices.max())
        
        image_shape = self.A.sel(unit_id=self.units[0]).values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx, cluster in enumerate(self.cluster_indices):
            final_image += np.stack((self.A.sel(unit_id=self.units[idx]).values,)*3, axis=-1) * viridis(cluster)[:3]
        
        return plt.imshow(final_image)
    
    def visualize_clusters_color(self):
        viridis = cm.get_cmap('viridis', len(np.unique(self.dendro["leaves_color_list"])))
        
        color_mapping= {}
        for i, leaf in enumerate(self.dendro['leaves']):
            color_mapping[leaf] = int(self.dendro['leaves_color_list'][i][1]) - 1 # Convert to int
        
        image_shape = self.A.sel(unit_id=self.units[0]).values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx in self.dendro['leaves']:
            final_image += np.stack((self.A.sel(unit_id=self.units[idx]).values,)*3, axis=-1) * viridis(color_mapping[idx])[:3]
        
        return plt.imshow(final_image)



class VisualizeGaussian():
    def __init__(self, C: xr.Dataset):
            self.current_id = C.coords["unit_id"].values[0]
            self.current_unit = C.sel(unit_id=self.current_id)
            
            # Stream
            Stream_unit = Stream.define("Stream_unit", unit=param.List())
            self.strm_unit = Stream_unit()
            self.strm_unit.add_subscriber(self.callback_unit)
            self.unit_sel = self.strm_unit.unit

            # Plot
            self.temp_comp = self._temp_comp(self.current_id)

            # Variable stuff
            self.variables = pn.widgets.Select(options=C.coords["unit_id"].values.tolist(), name='Unit ID', value=self.current_id)
            self.variables.param.watch(self._update_unit, 'value')

            self.slider = pn.widgets.FloatSlider(name='Gaussian Sigma', start=0, end=1, step=0.01)
            self.button = pn.widgets.Button(name='Apply Gaussian Smoothing', button_type='primary')


    def show(self) -> hv.HoloMap:
        return pn.Row(pn.Column(self.variables, self.slider, self.button), self.temp_comp)
    
    def callback_unit(self, unit=None):
        self.current_id = int(unit)
        self.current_unit = self.C.sel(unit_id=self.current_id)
        self._update_temp_comp(unit)

    def _update_temp_comp(self, unit=None):
        self.current_id = int(unit)
        self.current_unit = self.C.sel(unit_id=self.current_id)
        self.temp_comp.object = self._temp_comp(unit).object
    
    def _update_unit(self, unit):
        self.strm_unit.object = self.strm_unit.event(unit=unit.new)
    
    def _temp_comp(self, unit=None):
        cur_temp = hv.Dataset(
                self.current_unit
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")

        return hv.DynamicMap(cur_temp)


def peak_detection(C: xr.Dataset, **kwargs):
    peaks, _ = find_peaks(C, **kwargs)
    plt.plot(C)
    plt.plot(peaks, C[peaks], "x")

    return plt.show()


class FeatureExploration:
    """
    The purpose of this class is to explore potential features that can be used
    for the clustering of cells.
    """
    def __init__(
        self,
        dpath: str,
    ):
        
        mouseID, day, session = match_information(dpath)
        mouse_path, video_path = match_path(dpath)
        behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + session + "_" + "behavior_ms.csv"),sep=',')
        self.time = behavior_data['Time Stamp (ms)']
        self.behavior_data = behavior_data.to_xarray()

        minian_path = os.path.join(dpath, "minian")
        data = open_minian(minian_path)
        data_types = ['A', 'C', 'S', 'E']
        self.data = {}
        for dt in data_types:            
            if dt in data:
                self.data[dt] = data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None
        
        self.data['unit_ids'] = self.data['C'].coords['unit_id'].values

    def total_calcium_events(self, unit: int):
        """
        Calculate the total number of calcium events for a given unit.
        """
        return self.data['E'].sel(unit_id=unit).max()

    def get_timestep_ALP(self, unit: int):
        """
        Return a list that contains contains the a list of the frames where
        the ALP occurs
        """
        res = self.data['Frame Number'].sel(ALP > 0)
        res = res.tolist()
        return res
    
    def get_timestep_IALP(self, unit: int):
        
        """
        Return a list that contains contains the a list of the frames where
        the IALP occurs
        """
        res = self.data['Frame Number'].sel(IALP > 0)
        res = res.tolist()
        return res
    
    def get_timestep_RNFS(self, unit: int):
        """
        Return a list that contains contains the a list of the frames where
        the RNFS occurs
        """
        res = self.data['Frame Number'].sel(RNFS > 0)
        res = res.tolist()
        return res

    def get_section(self, starting_frame: int, duration: float, type: str = "C") -> xr.Dataset:
        """
        Return the selection of the data that is within the given time frame.
        duration indicates the number of frames.
        """
        # duration is in seconds convert to ms
        duration = duration * 1000
        start = self.time[starting_frame]
        end = start + duration
        frame_gap = 1
        while self.time[starting_frame + frame_gap] - self.time[starting_frame] < duration:
            frame_gap += 1


        if type in self.data:
            return self.data[type].sel(frame=slice(starting_frame, starting_frame+frame_gap))
        else:
            print("No %s data found in minian file" % (type))
            return None
    
    def get_AUC(self, section: xr.Dataset):
        """
        Calculate the area under the curve for a given section. Across all cells
        """
        if section.name is not "S":
            print("Invalid section type. Please use S not %s" % (section.name))
            return None

        return xr.apply_ufunc(
            np.sum,
            section.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[section.dtype],
        )
    
    def get_amplitude(self, section_signal: xr.Dataset, section_event: xr.Dataset):
        """
        Calculate the amplitude of the calcium event for a given section. Across all cells
        """
        if section_signal.name is not "S":
            print("Invalid section type. Please use S not %s" % (section_signal.name))
            return None
        if section_event.name is not "E":
            print("Invalid section type. Please use S not %s" % (section_event.name))
            return None

        all_cell_amplitudes = {}

        for unit_id in self.data['unit_ids']:
            cell_amplitudes = {}
            signal = section_signal.sel(unit_id=unit_id)
            event = section_event.sel(unit_id=unit_id)
            unique_events = np.unique(event)
            for event_id in unique_events:
                if event_id == 0:
                    continue
                cell_amplitudes[event_id] = np.sum(signal.where(event == event_id))
            all_cell_amplitudes[unit_id] = cell_amplitudes
        
        return all_cell_amplitudes
    
    def get_frequency(self, section: xr.Dataset):
        """
        Calculate the frequency of the calcium events for a given section. Across all cells
        """
        if section.name is not "E":
            print("Invalid section type. Please use S not %s" % (section.name))
            return None

        return xr.apply_ufunc(
            self.count_events,
            section.chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[section.dtype],
        )

    
    def count_events(self, a: np.ndarray) -> np.ndarray:
        """
        count the number of events in a given array.
        We do -1 to compensate for the 0 in the array
        """
        return np.unique(a).size - 1
    

