
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from post_minian.postprocessing import Feature, CellClustering
from panel.widgets import MultiSelect, StaticText, Select, Button, IntSlider, RangeSlider
from panel import Row, Column
from panel.layout import WidgetBox
import xarray as xr
import holoviews as hv
import panel as pn
from holoviews.streams import Stream
import param
from minian.visualization import centroid
from dask.diagnostics import ProgressBar

def plot_multiple_traces(explorer, neurons_to_plot=None, data_type='C', shift_amount=0.4, figure_ax = None):
    if figure_ax == None:
        fig, ax = plt.subplots(figsize=(40,len(neurons_to_plot)*0.5))
    else:
        ax = figure_ax
    if data_type in ['C', 'S', 'E', 'smoothed_C','filted_C']:
        if neurons_to_plot is None:
            print("Please specify which, neurons to plot")
            return None
        shifts = [shift_amount * i for i in range(len(neurons_to_plot))]
        for shift, neuron in zip(shifts, neurons_to_plot):
            trace = explorer.data[data_type].sel(unit_id = neuron)
            trace /= np.max(trace)
    #         ax.autoscale()
            ax.text(explorer.data['Time Stamp (ms)'][0],shift,neuron)
            ax.plot(explorer.data['Time Stamp (ms)'],trace + shift,alpha=0.5)
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('RNFS')],0,shifts[-1] + 1,color="green")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('IALP')],0,shifts[-1] + 1,color="blue")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('ALP')],0,shifts[-1] + 1,color="red",alpha=0.75)
    else:
        trace = explorer.data[data_type]
        max_trace = np.max(trace)
        ax.plot(explorer.data['Time Stamp (ms)'],trace,alpha=0.5)
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('RNFS')],0,max_trace,color = "green")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('IALP')],0,max_trace,color = "blue")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('ALP')],0,max_trace,color = "red",alpha=0.75)
    return ax
    #fig.savefig(os.path.join("/N/project/Cortical_Calcium_Image/analysis", mouseID+'_'+day+'_'+session+"_trace_test_ms.pdf"))


def plot_multiple_traces_segment(segment, neurons_to_plot, shift_amount=0.4):
    shifts = [shift_amount * i for i in range(len(neurons_to_plot))]
    fig, ax = plt.subplots(figsize=(15,5))
    for shift, neuron in zip(shifts, neurons_to_plot):
        trace = segment.sel(unit_id = neuron)
        trace /= np.max(trace)
#         ax.autoscale()
        ax.plot(segment['frame'],trace + shift,alpha=0.5)
    fig.savefig(os.path.join("/N/project/Cortical_Calcium_Image/analysis", mouseID+'_'+day+'_'+session+"_trace_test_ms.pdf"))


class ClusteringExplorer:
    """
    Interactive visualization for clustering results.
    
    TODO add more details
    """

    def __init__(
        self,
        dataset: List[object],
        events: Optional[pd.DataFrame] = None,

    ):
        self.data = {}
        self.features = {}
        for element in dataset:
            if isinstance(element, xr.DataArray):
                self.data[element.name] = element
                if element.name == "A":
                    with ProgressBar():
                        self.Asum = element.sum("unit_id").compute()
            
            else:
                print(f"Warning: {type(element)} is not an allowed data type and will be ignored.")
        
        self._all_cells = None
        self.cell_clustering = None
        if events is not None:
            self.events = events


        # Streams
        Stream_usub = Stream.define("Stream_usub", usub=param.Integer())
        self.strm_usub = Stream_usub()
        self.strm_usub.add_subscriber(self.callback_usub)
        self.usub_sel = self.strm_usub.usub

        # Widgets
        self.widgets = self._create_widgets()

    
    def callback_usub(self, usub=None):
        self.update_temp_comp_sub(usub)

    def _create_widgets(self):
        """
        Widgets associated with the initial values loaded and their description.
        """
        # Implement multiselect for features, this will occupy the left side of the panel
        self.currents_events = {}
        w_data_select = MultiSelect(name='Data Selection',
                options=list(self.data.keys()))
        
        #adding features that are TRUE in MultiSelect
        w_added_feature_select = MultiSelect(name='Loaded Features')
        
        # Utility widgets
        w_event_filter_select = MultiSelect(name='Event Filter', options=['ALP','IALP','RNFS'])
        w_initial_range = RangeSlider(name='Specify Range', start=0, end=100, value=(0,100), step=1)
        w_haste_slider = IntSlider(name="Haste", value=0,step=1,start=0,end=30)
        w_window_slider = IntSlider(name="Window Size", value=1,step=1,start=0,end=30)


        # Display information from selected feature
        w_select_cell = Select(name='Select Cell', options=[])
        self.w_visualize = pn.panel(hv.Curve([]).opts(xaxis=None,yaxis=None,xlabel=None,ylabel=None), width=400, height=400, sizing_mode="scale_both")
        
        self.w_visualize_dendogram = pn.pane.Matplotlib(width=400, height=200)
        self.w_visualize_cluster = pn.pane.Matplotlib(width=400, height=200)
        
        self.w_description = StaticText(name="Description", value="")
        self.w_ranges = StaticText(name="Ranges", value="")
        self.w_events = StaticText(name="Events", value="")
        self.w_cluster_distance = IntSlider(name="Cluster Distance", value=1,step=1,start=1,end=1000)
        w_distance_metric = Select(name='Distance Metrics', options=['euclidean', 'cosine', 'manhattan'])

        def update_usub(usub):
            to_int = int(usub.new.split(" ")[1])
            self.strm_usub.event(usub=to_int)
        
        def update_feature_info(event=None):
            self.currents_events = {}
            self.ranges = None
            if event is None:
                self.main_panel.objects = pn.Row(self.pn_data_features).objects
            else:
                selected_features = event.new
                if selected_features:
                    selected_feature_name = selected_features[0]
                    selected_feature = self.data.get(selected_feature_name)
                    # Visualization stuff for C ,E and S
                    if selected_feature is not None:
                        if selected_feature.name in ['C','S','E']:
                            self.selection = selected_feature
                            self._all_cells = self.selection.isel(frame=0).dropna("unit_id").coords["unit_id"].values
                            w_select_cell.options = [f"Cell {u}" for u in self._all_cells]
                            self.update_temp_comp_sub(self._all_cells[0])
                            if self.pn_data_features.active == 0:
                                w_initial_range.end = self.selection.shape[1]
                                w_initial_range.value = (0, w_initial_range.end)
                                self.main_panel.objects = pn.Row(self.pn_data_features, self.pn_utility, pn.Tabs(self.pn_description)).objects
                            else:
                                self.main_panel.objects = pn.Row(self.pn_data_features, pn.Tabs(self.pn_description_advanced, self.pn_dendrogram, self.pn_clustering)).objects
                    
                        elif selected_feature.name == 'A':
                            self.w_visualize.object = hv.Image(self.Asum)

                            if self.pn_data_features.active == 0:
                                self.main_panel.objects = pn.Row(self.pn_data_features, self.pn_utility, pn.Tabs(self.pn_description)).objects
                            else:
                                self.main_panel.objects = pn.Row(self.pn_data_features, pn.Tabs(self.pn_description, self.pn_dendrogram, self.pn_clustering)).objects
                    else:
                        if selected_feature_name in self.features.keys():
                            self.w_description.value = self.features[selected_feature_name].description
                            self.w_ranges.value = self.features[selected_feature_name].ranges
                            
                            '''
                                Implementing the visualization of selected features.  
                            '''
                            #self.selection = self.features[selected_feature_name].values
                            #self._all_cells = self.selection.dropna("unit_frame").coords["unit_id"].values
                            #w_select_cell.options = [f"Cell {u}" for u in self._all_cells]
                            #self.update_temp_comp_sub(self._all_cells[0])
                            
                            self.main_panel.objects = pn.Row(self.pn_data_features, pn.Tabs(self.pn_description_advanced, self.pn_dendrogram, self.pn_clustering)).objects
                            

        def load_feature(clicks=None):
            if w_data_select.value:
                w_added_feature_select.options = w_added_feature_select.options + w_data_select.value
        
        def unload_feature(clicks=None):
            if w_added_feature_select.value:
                self.features.pop(w_added_feature_select.value[0])
                trimmed_options = [option for option in w_added_feature_select.options if option not in w_added_feature_select.value]
                w_added_feature_select.options = trimmed_options
                #print(self.features)
        
        def load_dendrogram(clicks=None):
            w_added_feature_select.options = ["C"]
            if w_added_feature_select.options:
                load_dendrogram_button.name = "Loading..."
                selected_feature = self.data.get(w_added_feature_select.options[0])
                self.cell_clustering = CellClustering(selected_feature, self.data['A'])
                fig, ax = plt.subplots()
                self.cell_clustering.visualize_dendrogram(ax=ax)
                self.w_visualize_dendogram.object = fig
                load_dendrogram_button.name = "Generate Dendrogram from Loaded Features"
        
        def load_cluster(clicks=None):
            w_added_feature_select.options = ["C"]
            if w_added_feature_select.options:
                load_cluster_button.name = "Loading..."
                selected_feature = self.data.get(w_added_feature_select.options[0])
                self.cell_clustering = CellClustering(selected_feature, self.data['A'], metric=w_distance_metric.value)
                fig, ax = plt.subplots()
                ax.imshow(self.cell_clustering.visualize_clusters(distance=self.w_cluster_distance.value))
                ax.set_axis_off()
                self.w_visualize_cluster.object = fig
                load_cluster_button.name = "Load Cluster from Loaded Features"
        
        def load_filter(clicks=None):
            trimmed_df = self.events.loc[self.ranges[0]:self.ranges[1],:]
            if w_event_filter_select.value:
                selected_events = w_event_filter_select.value
                values = []
                for event in selected_events:
                    pre_trimmed_list = trimmed_df.index[trimmed_df[event] > 0].to_list()
                    for index in pre_trimmed_list:
                        values.append(self.get_section(index - w_haste_slider.value, w_window_slider.value))
                
                name = ','.join(w_data_select.value) + ":" + ','.join(selected_events) + '  ' + str(w_initial_range.value)
                description = (f"{name} contains all the values from {self.ranges[0]} to {self.ranges[1]} "
                               f"filtered by {','.join(selected_events)} with a window of {w_window_slider.value} "
                               f"and a haste of {w_haste_slider.value}.")
            else:
                name = ','.join(w_data_select.value)
                timeframevalue = tuple(self.ranges)
                values = trimmed_df
                description = f"{name} contains all the values from {timeframevalue[0]} to {timeframevalue[1]}"
            
            # Merge the list of xarrays along the 'unit_id' dimension
            merged_data = xr.concat(values, dim='unit_id')

            # Flatten the merged xarray along the 'frame' dimension
            flattened_data = merged_data.stack(unit_frame=('frame', 'unit_id'))
            
            self.features[name] = Feature(name=name, ranges=self.ranges, values=flattened_data, description=description)
            #print('Here is the feaeture:',flattened_data) #Bug Print
            #print('Cell Data: ',flattened_data.dropna("unit_frame").coords["unit_id"].values)
            w_added_feature_select.options = w_added_feature_select.options + [name]
        
        def data_feat_switch(event):
            w_data_select.value = []
            w_added_feature_select.value = []
            update_feature_info()

        def update_visual_ranges(event):
            self.ranges = event.new
            self.update_temp_comp_sub(None)
        
        def update_visual_events(event):
            events = event.new
            self.currents_events = {}

            for event in events:
                pre_trimmed_list = self.events.index[self.events[event] > 0].to_list()

                remove_distance = 0
                new_list = []

                for i, val in enumerate(pre_trimmed_list):
                    if i == 0:
                        new_list.append(val)
                    else:
                        if val - pre_trimmed_list[i-1] > remove_distance:
                            new_list.append(val)
                
                if new_list:
                    self.currents_events[event] = hv.VLine(new_list[0]).opts(color='green')
                    for i in range(1,  len(new_list)):
                        self.currents_events[event] *= hv.VLine(new_list[i]).opts(color='green')
                
            self.update_temp_comp_sub(None)



                 
        #adding buttons 
        load_feature_button = Button(name='Load', button_type='success')
        unload_feature_button = Button(name='Unload', button_type='danger')
        filter_button = Button(name='Filter', button_type='success')
        
        load_dendrogram_button = Button(name='Generate Dendrogram from Loaded Features', button_type='primary')
        load_cluster_button = Button(name='Load Cluster from Loaded Features', button_type='primary')
        
        load_feature_button.param.watch(load_feature, "clicks")
        unload_feature_button.param.watch(unload_feature, "clicks")
        load_dendrogram_button.param.watch(load_dendrogram, "clicks")
        load_cluster_button.param.watch(load_cluster,"clicks")
        filter_button.param.watch(load_filter,"clicks")
        w_select_cell.param.watch(update_usub, "value")
        w_initial_range.param.watch(update_visual_ranges, "value")
        w_event_filter_select.param.watch(update_visual_events, "value")
        
        # Register the callback with the value attribute of the feature selection widget               
        self.pn_data_features = pn.Tabs(('Initial Data',w_data_select),('Loaded Features',Column(w_added_feature_select,unload_feature_button)))
        self.pn_utility = Column(w_initial_range,
                                 w_event_filter_select,
                                 w_haste_slider,
                                 w_window_slider,
                                 filter_button, width=300)
        self.pn_description = ('Description',Column(self.w_visualize, w_select_cell))
        self.pn_description_advanced = ('Description',Column(self.w_visualize, w_select_cell, w_distance_metric, self.w_description, self.w_ranges))
        self.pn_dendrogram = ('Dendrogram',Column(load_dendrogram_button, self.w_visualize_dendogram))
        self.pn_clustering = ('Cluster',Column(self.w_cluster_distance, load_cluster_button, self.w_visualize_cluster))
        
        self.pn_data_features.param.watch(data_feat_switch, 'active')
        self.main_panel = pn.Row(self.pn_data_features)

        w_data_select.param.watch(update_feature_info, 'value')
        w_added_feature_select.param.watch(update_feature_info, 'value')

        w_data_select.value = ['S']
        w_event_filter_select.value = ['RNFS', 'IALP']
        w_window_slider.value = 10
        load_filter()
        

    def _temp_comp_sub(self, usub=None, data=None, ranges=None):
        if usub is None:
            usub = self.strm_usub.usub
        
        signal = hv.Dataset(
                data.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        
        if ranges is not None:
            signal *= hv.VLine(ranges[0]).opts(color='red') * hv.VLine(ranges[1]).opts(color='red')
        for visual_events in self.currents_events.values():
            signal *= visual_events
        return pn.panel(signal, width=400, height=200)
        
    
    
    def update_temp_comp_sub(self, usub=None):
        self.w_visualize.object = self._temp_comp_sub(usub, self.selection, self.ranges).object

    def get_section(self, starting_frame: int, duration: float) -> xr.Dataset:
        """
        Return the selection of the data that is within the given time frame.
        duration indicates the number of frames.
        """
        # duration is in seconds convert to ms
        duration *= 1000
        max_length = len(self.events.index)
        frame_gap = 1
        while self.events['Time Stamp (ms)'][starting_frame + frame_gap] - self.events['Time Stamp (ms)'][starting_frame] < duration and starting_frame + frame_gap < max_length:
            frame_gap += 1

        
        return self.selection.sel(frame=slice(starting_frame, starting_frame+frame_gap))
    
    def show(self) -> Row:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolboxes.
        """
        return self.main_panel

    def display(self):
        self.show().show()