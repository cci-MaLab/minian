from typing import Optional, Union, List
import xarray as xr
import pandas as pd
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import cm
from matplotlib import colors 
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews.util import Dynamic
import panel as pn
from scipy.ndimage import gaussian_filter1d
from holoviews.streams import Stream
import param
from scipy.signal import find_peaks
import os
from scipy.ndimage.measurements import center_of_mass
import dask.array as da
from dask.diagnostics import ProgressBar

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
        section: Optional[xr.Dataset] = None,
        A: Optional[xr.DataArray] = None,
        fft: bool = True,
        metric: Optional[str] = 'euclidean'
    ):
    
        self.signals = section
        self.A = A
        self.units = self.signals.isel(frame=0).dropna("unit_id").coords["unit_id"].values
        self.psd_list = []

        if fft:
            for unit in self.units:
                self.compute_psd(unit) # compute psd for each unit
        else:
            self.psd_list = [self.signals.sel(unit_id=unit).values for unit in self.units]
        
        # Compute agglomerative clustering
        self.linkage_data = linkage(self.psd_list, method='average', metric=metric)

    def compute_psd(self, unit: int):
        val = self.signals.sel(unit_id=unit).values
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list.append(psd)
    
    def visualize_dendrogram(self, color_threshold=None, ax=None):
        self.dendro = dendrogram(self.linkage_data, color_threshold=color_threshold, ax=ax)
        return self.dendro


    def visualize_clusters(self, distance):
        self.cluster_indices = fcluster(self.linkage_data, distance, criterion='distance')
        viridis = cm.get_cmap('viridis', self.cluster_indices.max())
        
        image_shape = self.A.sel(unit_id=self.units[0]).values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx, cluster in enumerate(self.cluster_indices):
            final_image += np.stack((self.A.sel(unit_id=self.units[idx]).values,)*3, axis=-1) * viridis(cluster)[:3]
        
        return final_image
    
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
        print(session)
        if (session is None):
            behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + "behavior_ms.csv"),sep=',')
        else:
            behavior_data = pd.read_csv(os.path.join(mouse_path, mouseID + "_" + day + "_" + session + "_" + "behavior_ms.csv"),sep=',')
        data_types = ['RNFS', 'ALP', 'IALP', 'Time Stamp (ms)']
        self.data = {}
        for dt in data_types:            
            if dt in behavior_data:
                self.data[dt] = behavior_data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None

        minian_path = os.path.join(dpath, "minian")
        data = open_minian(minian_path)
        data_types = ['A', 'C', 'S', 'E']
        for dt in data_types:            
            if dt in data:
                self.data[dt] = data[dt]
            else:
                print("No %s data found in minian file" % (dt))
                self.data[dt] = None
        
        self.data['unit_ids'] = self.data['C'].coords['unit_id'].values
        self.dpath = dpath
        self.data['collapsed_E'] = None

        output_dpath = "/N/project/Cortical_Calcium_Image/analysis"
        if session is None:
            self.output_path = os.path.join(output_dpath, mouseID,day)
        else:
            self.output_path = os.path.join(output_dpath, mouseID,day,session)

        if(os.path.exists(self.output_path) == False):
            os.makedirs(self.output_path)

    def total_calcium_events(self, unit: int):
        """
        Calculate the total number of calcium events for a given unit.
        """
        return self.data['E'].sel(unit_id=unit).max().values.item()

    def get_timestep(self, type: str):
        """
        Return a list that contains contains the a list of the frames where
        the ALP occurs
        """
        return np.flatnonzero(self.data[type])

    def get_section(self, starting_frame: int, duration: float, delay: float = 0.0, include_prior: bool = False, type: str = "C") -> xr.Dataset:
        """
        Return the selection of the data that is within the given time frame.
        duration indicates the number of frames.
        """
        # duration is in seconds convert to ms
        duration *= 1000
        delay *= 1000
        start = self.data['Time Stamp (ms)'][starting_frame]
        max_length = len(self.data['Time Stamp (ms)'])
        if delay > 0:
            frame_gap = 1
            while self.data['Time Stamp (ms)'][starting_frame + frame_gap] - self.data['Time Stamp (ms)'][starting_frame] < delay:
                frame_gap += 1
            starting_frame += frame_gap
        if include_prior:
            frame_gap = -1
            while self.data['Time Stamp (ms)'][starting_frame] - self.data['Time Stamp (ms)'][starting_frame + frame_gap] < duration and starting_frame + frame_gap > 0:
                frame_gap -= 1
            starting_frame += frame_gap
            duration *= 2
        frame_gap = 1
        while self.data['Time Stamp (ms)'][starting_frame + frame_gap] - self.data['Time Stamp (ms)'][starting_frame] < duration and starting_frame + frame_gap < max_length:
            frame_gap += 1


        if type in self.data:
            return self.data[type].sel(frame=slice(starting_frame, starting_frame+frame_gap))
        else:
            print("No %s data found in minian file" % (type))
            return None
    
    def get_AUC(self, section: xr.Dataset, section_event: xr.Dataset):
        """
        Calculate the area under the curve for a given section. Across all cells
        """
        if section.name != "S":
            print("Invalid section type. Please use S not %s" % (section.name))
            return None

        amplitudes = self.get_amplitude(section, section_event)
        total_auc = {}
        for name, cell_events in amplitudes.items():
            total_auc[name] = 0
            for event_name, auc in cell_events.items():
                total_auc[name] += auc

        return total_auc
    
    def get_amplitude(self, section_signal: xr.Dataset, section_event: xr.Dataset):
        """
        Calculate the amplitude of the calcium event for a given section. Across all cells
        """
        if section_signal.name != "S":
            print("Invalid section type. Please use S not %s" % (section_signal.name))
            return None
        if section_event.name != "E":
            print("Invalid section type. Please use S not %s" % (section_event.name))
            return None

        all_cell_amplitudes = {}

        for unit_id in self.data['unit_ids']:
            cell_amplitudes = {}
            signal = section_signal.sel(unit_id=unit_id).values
            event = section_event.sel(unit_id=unit_id).values
            unique_events = np.unique(event)
            for event_id in unique_events:
                if event_id == 0:
                    continue
                cell_amplitudes[event_id] = np.sum(signal[event == event_id])
            all_cell_amplitudes[unit_id] = cell_amplitudes
        
        return all_cell_amplitudes
    
    def get_frequency(self, section: xr.Dataset, time: float):
        """
        Calculate the frequency of the calcium events for a given section. Across all cells
        """
        if section.name != "E":
            print("Invalid section type. Please use S not %s" % (section.name))
            return None

        all_cell_frequency = {}

        for unit_id in self.data['unit_ids']:
            cell_frequency = {}
            event = section.sel(unit_id=unit_id).values
            unique_events = np.unique(event)
            all_cell_frequency[unit_id] = len(unique_events)-1 / time
        return all_cell_frequency

    
    def count_events(self, a: np.ndarray) -> np.ndarray:
        """
        count the number of events in a given array.
        We do -1 to compensate for the 0 in the array
        """
        return np.unique(a).size - 1

        
    def collapse_E_events(self, smoothing="gauss", kwargs=None) -> None:
        """
        Collapse the E values by summing up the values.
        """
        non_collapsed_E = xr.apply_ufunc(
            self.normalize_events,
            self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        )

        self.data['collapsed_E'] = non_collapsed_E.sum(dim='unit_id')

        if smoothing == "gauss":
            self.data['collapsed_E'] = xr.apply_ufunc(
                gaussian_filter1d,
                self.data['collapsed_E'],
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                dask="parallelized",
                kwargs=kwargs,
                output_dtypes=[self.data['E'].dtype]
            )
        elif smoothing == "mean":
            self.data['collapsed_E'] = xr.apply_ufunc(
                self.moving_average,
                self.data['collapsed_E'],
                input_core_dims=[["frame"]],
                output_core_dims=[["frame"]],
                dask="parallelized",
                kwargs=kwargs,
                output_dtypes=[self.data['E'].dtype]
            ).compute()
    
    def collapse_E_events_AUC(self) -> None:
        """
        Collapse the E values by summing up the values.
        """
        non_collapsed_E = xr.apply_ufunc(
            self.normalize_events,
            self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        )

        non_collapsed_E *= self.data['S']

        self.data['collapsed_E_AUC'] = non_collapsed_E.sum(dim='unit_id')

    def moving_average(self, x, w=100, type='constant'):
        return np.convolve(x, np.ones(w), type) / w

    def collapse_E_events_peak(self) -> None:
        '''
        Get the events' peak
        '''
        non_collapsed_E_peak = xr.apply_ufunc(
            self.derivative,
            self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        ).compute()
        self.data['collapsed_E_peak'] = non_collapsed_E_peak.sum(dim='unit_id')

    def derivative(self, a: np.ndarray) -> np.ndarray:
        a = a.copy()
        b = np.roll(a, 1, axis=1)
        # b[:, 0] = 0
        c = a - b
        c[c > 0] = 0
        c[c < 0] = 1
        c = np.roll(c, -1, axis=1)
        return c



    def find_events_peak(self, a: np.ndarray) -> np.ndarray:
        a = a.copy()
        print(a.shape)
        res = np.zeros(np.shape(a))
        for row,b in enumerate(a):
            u,i,c=np.unique(b,return_index = True,return_counts = True)
            for m in range(1,len(u)):                
                res[row,(i[m]+c[m]-1)]=u[m]
        return res


    def normalize_events(self, a: np.ndarray) -> np.ndarray:
        """
        All positive values are converted to 1.
        """
        a = a.copy()
        a[a > 0] = 1
        return a

    
    def centroid(self, verbose=False) -> pd.DataFrame:
        """
        Compute centroids of spatial footprint of each cell.

        Parameters
        ----------
        A : xr.DataArray
            Input spatial footprints.
        verbose : bool, optional
            Whether to print message and progress bar. By default `False`.

        Returns
        -------
        cents_df : pd.DataFrame
            Centroid of spatial footprints for each cell. Has columns "unit_id",
            "height", "width" and any other additional metadata dimension.
        """
        A = self.data['A']
        def rel_cent(im):
            im_nan = np.isnan(im)
            if im_nan.all():
                return np.array([np.nan, np.nan])
            if im_nan.any():
                im = np.nan_to_num(im)
            cent = np.array(center_of_mass(im))
            return cent / im.shape

        gu_rel_cent = da.gufunc(
            rel_cent,
            signature="(h,w)->(d)",
            output_dtypes=float,
            output_sizes=dict(d=2),
            vectorize=True,
        )
        cents = xr.apply_ufunc(
            gu_rel_cent,
            A.chunk(dict(height=-1, width=-1)),
            input_core_dims=[["height", "width"]],
            output_core_dims=[["dim"]],
            dask="allowed",
        ).assign_coords(dim=["height", "width"])
        if verbose:
            print("computing centroids")
            with ProgressBar():
                cents = cents.compute()
        cents_df = (
            cents.rename("cents")
            .to_series()
            .dropna()
            .unstack("dim")
            .rename_axis(None, axis="columns")
            .reset_index()
        )
        h_rg = (A.coords["height"].min().values, A.coords["height"].max().values)
        w_rg = (A.coords["width"].min().values, A.coords["width"].max().values)
        cents_df["height"] = cents_df["height"] * (h_rg[1] - h_rg[0]) + h_rg[0]
        cents_df["width"] = cents_df["width"] * (w_rg[1] - w_rg[0]) + w_rg[0]
        return cents_df

    def get_filted_C(self) -> None:
        non_collapsed_E = xr.apply_ufunc(
            self.normalize_events,
            self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        )
        filted_C = self.data['C'] * non_collapsed_E
        self.data['filted_C'] = filted_C

    def smoothed_C(self,window_length = 6, n = 3,mode = "savgol",unit_id = []) -> None:
        self.data['smoothed_C'] = self.data['C']
        if unit_id is None:
            unit_id = self.data['unit_ids']
        if mode =="savgol":
            smoothed_C = xr.apply_ufunc(
                savgol_filter,
                self.data['C'].sel(unit_id=unit_id),
                window_length,
                n,
                input_core_dims=[["frame"],[],[]],
                output_core_dims=[["frame"]],
                dask="parallelized",
                output_dtypes=[self.data['C'].dtype],
            )
        elif mode =="gauss":
            smoothed_C = xr.apply_ufunc(
                gaussian_filter1d,
                self.data['C'].sel(unit_id=unit_id),
                3,
                input_core_dims=[["frame"],[]],
                output_core_dims=[["frame"]],
                dask="parallelized",
                output_dtypes=[self.data['C'].dtype],
            )
        for uid in unit_id:
            self.data['smoothed_C'].sel(unit_id = uid).values = smoothed_C.sel(unit_id=uid)

    def smoothed_filted_C(self) -> None:
        non_collapsed_E = xr.apply_ufunc(
            self.normalize_events,
            self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
            input_core_dims=[["frame"]],
            output_core_dims=[["frame"]],
            dask="parallelized",
            output_dtypes=[self.data['E'].dtype],
        )
        smoothed_filted_C = self.data['smoothed_C'] * non_collapsed_E
        self.data['smoothed_filted_C'] = smoothed_filted_C

    # def calculate_slop(self, a: np.ndarray) -> None:
    #     a = a.copy()
    #     a

    # def get_events_rising_part(self) -> None:
    #     #calculate_slop
    #     calculate_E = xr.apply_ufunc(
    #         self.calculate_slop,
    #         self.data['E'].chunk(dict(frame=-1, unit_id="auto")),
    #         input_core_dims=[["frame"]],
    #         output_core_dims=[["frame"]],
    #         dask="parallelized",
    #         output_dtypes=[self.data['E'].dtype],
    #     )
    #     smoothed_filted_C = self.data['smoothed_C'] * non_collapsed_E
    #     self.data['smoothed_filted_C'] = smoothed_filted_C



class Feature:
    '''
        Parameters
        ----------
        name : str
        timeframevalue : tuple 
        values : numpy array 
        description : str
        dist_met : str, optional
            Distance metrics is method of finding distance. By default its Euclidean
        event :  str, optional
            event can be ALP/IALP/RNFS
    '''

    def __init__(self,
                name: str, 
                ranges: tuple, 
                values: Union[xr.DataArray, List[xr.DataArray], xr.Dataset],  
                description: str,
                dist_met: Optional[str] = None, 
                event: Union[str, List[str], None] = None
        ):
        self.name = name
        self.ranges = ranges
        self.values = values
        self.dist_met = dist_met
        self.description = description
        self.event = event


class NewFeatures:
    '''
        Parameters
        ----------
        ALP
    '''

    def __init__(self,
                A: dict,
                ALP: List[xr.DataArray],
                IALP: List[xr.DataArray],
                RNFS: List[xr.DataArray],  
                events: Optional[List[str]] = None, 
                description: Optional[str] = None,
                dist_met: Optional[str] = None, 
        ):
        self.A = A
        self.ALPlist = ALP
        self.IALPlist = IALP
        self.RNFSlist = RNFS
        self.events = events
        self.dist_met = dist_met
        self.timefilter = None
        self.description = description
        self.set_timefilter()
        self.set_vector(self.events)

    
    def set_timefilter(self):
        ALP = {}
        IALP = {}
        RNFS = {}
        if self.timefilter is None:
            for i in self.ALPlist:
                for j in i.coords['unit_id'].values:
                    try:
                        ALP[j]
                    except:
                        ALP[j] = np.array([])
                    ALP[j] = np.r_['-1', ALP[j], np.array(i.sel(unit_id=j).values)]
            for i in self.IALPlist:
                for j in i.coords['unit_id'].values:
                    try:
                        IALP[j]
                    except:
                        IALP[j] = np.array([])
                    IALP[j] = np.r_['-1', IALP[j], np.array(i.sel(unit_id=j).values)]
            for i in self.RNFSlist:
                for j in i.coords['unit_id'].values:
                    try:
                        RNFS[j]
                    except:
                        RNFS[j] = np.array([])
                    RNFS[j] = np.r_['-1', RNFS[j], np.array(i.sel(unit_id=j).values)]
        self.ALP = ALP
        self.IALP = IALP
        self.RNFS = RNFS
        self.set_vector(self.events)


    def set_events(self, events:List[str]):
        self.events = events
        self.set_vector(self.events)

    def set_vector(self, events:list):
        '''
        event :  str, list
            event can be ALP/IALP/RNFS
        '''
        if events is None:
            events=['ALP','IALP','RNFS']
        values = {}
        if 'ALP' in events:
            for key in self.ALP:
                try:
                    values[key]
                except:
                    values[key] = np.array([])
                values[key] = np.r_['-1', values[key], self.ALP[key]]            
        if 'IALP' in events:
            for key in self.IALP:
                try:
                    values[key]
                except:
                    values[key] = np.array([])
                values[key] = np.r_['-1', values[key], self.IALP[key]]
        if 'RNFS' in events:
            for key in self.RNFS:
                try:
                    values[key]
                except:
                    values[key] = np.array([])
                values[key] = np.r_['-1', values[key], self.RNFS[key]]
        if values == np.array([]):
            values
        self.values = values
    
    def set_description(self, content:str):
        self.description = content

    def reset_dataArray(self, a: np.ndarray):
        a = np.array([])
        return a
    

class NewCellClustering:
    """
    Cell clustering class. This class is used to cluster cells based on their
    temporal activity, using FFT and agglomerative clustering.
    """

    def __init__(
        self,
        section: Optional[dict] = None,
        A: Optional[xr.DataArray] = None,
        fft: bool = True
    ):
        self.A = A
        self.signals = section
        self.psd_list = []

        if fft:
            for unit_id in self.signals:
                self.compute_psd(unit_id) # compute psd for each unit
        else:
            self.psd_list = [self.signals[unit_id] for unit_id in self.signals]        
        # Compute agglomerative clustering
        self.linkage_data = linkage(self.psd_list, method='average', metric='euclidean')

    def compute_psd(self, unit: int):
        val = self.signals[unit]
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list.append(psd)
    
    def visualize_dendrogram(self, color_threshold=None, ax=None):
        self.dendro = dendrogram(self.linkage_data,labels=list(self.signals.keys()), color_threshold=color_threshold, ax=ax)
        return self.dendro

    def visualize_clusters(self, t=4):
        self.cluster_indices = fcluster(self.linkage_data, t, criterion='maxclust')
        
        viridis = cm.get_cmap('jet', self.cluster_indices.max()+1)

        
        image_shape = self.A[list(self.A.keys())[0]].values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))
        # print((self.A[list(self.A.keys())[0]].values,)*3)
        for idx, cluster in enumerate(self.cluster_indices):
           
           
            final_image += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(cluster)[:3]
        
        return final_image
    
    def visualize_clusters_color(self):
        viridis = cm.get_cmap('viridis', len(np.unique(self.dendro["leaves_color_list"])))
        
        color_mapping= {}
        for i, leaf in enumerate(self.dendro['leaves']):
            color_mapping[leaf] = int(self.dendro['leaves_color_list'][i][1]) - 1 # Convert to int
        
        image_shape = self.A[list(self.A.keys())[0]].values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx in self.dendro['leaves']:
            final_image += np.stack((self.A[list(self.A.keys())[idx]].values,)*3, axis=-1) * viridis(color_mapping[idx])[:3]
        
        return plt.imshow(final_image)