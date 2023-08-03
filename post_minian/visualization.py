
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from post_minian.postprocessing import Feature, CellClustering
from panel.widgets import MultiSelect, StaticText, Select, Button, IntSlider
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
        fig, ax = plt.subplots(figsize=(40,5))
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
        dataset: List[object]
    ):
        self.data = {}
        for element in dataset:
            if isinstance(element, xr.DataArray):
                self.data[element.name] = element
                if element.name == "A":
                    with ProgressBar():
                        self.Asum = element.sum("unit_id").compute()
            
            else:
                print(f"Warning: object {type(element)} is not an allowed data type and will be ignored.")
        
        self._all_cells = None
        self.cell_clustering = None


        # Streams
        Stream_usub = Stream.define("Stream_usub", usub=param.Integer())
        self.strm_usub = Stream_usub()
        self.strm_usub.add_subscriber(self.callback_usub)
        self.usub_sel = self.strm_usub.usub

        # Widgets
        self.widgets = self._description_widgets()

    
    def callback_usub(self, usub=None):
        self.update_temp_comp_sub(usub)

    def _description_widgets(self):
        """
        Widgets associated with the initial values loaded and their description.
        """
        # Implement multiselect for features, this will occupy the left side of the panel
        w_data_select = MultiSelect(name='Data Selection',
                options=list(self.data.keys()))
        
        #adding features that are TRUE in MultiSelect
        w_added_feature_select = MultiSelect(name='Loaded Features')
        
        #adding filter 
        w_event_filter_select = MultiSelect(name='Event Filter', options=['ALP','IALP','RNFS'])
        
        # Display information from selected feature
        w_select_cell = Select(name='Select Cell', options=[])
        self.w_visualize = pn.panel(hv.Curve([]).opts(xaxis=None,yaxis=None,xlabel=None,ylabel=None), width=400, height=400, sizing_mode="scale_both")
        
        self.w_visualize_dendogram = pn.pane.Matplotlib(width=400, height=200)
        self.w_visualize_cluster = pn.pane.Matplotlib(width=400, height=200)
        
        w_description = StaticText(name="Description", value="")
        w_ranges = StaticText(name="Ranges", value="")
        w_events = StaticText(name="Events", value="")
        self.w_cluster_distance = IntSlider(name="Cluster Distance", value=1,step=1,start=1,end=1000)
        self.w_time_filter = IntSlider(name="Time Filter", value=1,step=1,start=1,end=30)
        w_distance_metric = Select(name='Select', options=['Euclidean', 'Cosine', 'Manhattan'])

        def update_usub(usub):
            to_int = int(usub.new.split(" ")[1])
            self.strm_usub.event(usub=to_int)

        w_select_cell.param.watch(update_usub, "value")
        
        def update_feature_info(event=None):
            if event is None:
                self.main_panel.objects = pn.Row(self.pn_data_features).objects
            else:
                selected_features = event.new
                if selected_features:
                    selected_feature_name = selected_features[0]
                    selected_feature = self.data.get(selected_feature_name)

                    # Visualization stuff for C and S
                    if selected_feature.name in ['C','S']:
                        self.selection = selected_feature
                        self._all_cells = self.selection.isel(frame=0).dropna("unit_id").coords["unit_id"].values
                        w_select_cell.options = [f"Cell {u}" for u in self._all_cells]
                        self.update_temp_comp_sub(self._all_cells[0])
                        if self.pn_data_features.active == 0:
                            self.main_panel.objects = pn.Row(self.pn_data_features, self.pn_utility, pn.Tabs(self.pn_description)).objects
                        else:
                            self.main_panel.objects = pn.Row(self.pn_data_features, pn.Tabs(self.pn_description_advanced, self.pn_dendrogram, self.pn_clustering)).objects
                
                    elif selected_feature.name == 'A':
                        self.w_visualize.object = hv.Image(self.Asum)

                        if self.pn_data_features.active == 0:
                            self.main_panel.objects = pn.Row(self.pn_data_features, self.pn_utility, pn.Tabs(self.pn_description)).objects
                        else:
                            self.main_panel.objects = pn.Row(self.pn_data_features, pn.Tabs(self.pn_description, self.pn_dendrogram, self.pn_clustering)).objects

        def load_feature(clicks=None):
            if w_data_select.value:
                w_added_feature_select.options = w_added_feature_select.options + w_data_select.value
        
        def unload_feature(clicks=None):
            if w_added_feature_select.value:
                trimmed_options = [option for option in w_added_feature_select.options if option not in w_added_feature_select.value]
                w_added_feature_select.options = trimmed_options
        
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
                self.cell_clustering = CellClustering(selected_feature, self.data['A'])
                fig, ax = plt.subplots()
                ax.imshow(self.cell_clustering.visualize_clusters(distance=self.w_cluster_distance.value))
                ax.set_axis_off()
                self.w_visualize_cluster.object = fig
                load_cluster_button.name = "Load Cluster from Loaded Features"
        
        def load_filter(clicks=None):
            #Functionality of Load_Feature Class 
            if w_data_select.value:
                w_added_feature_select.options = w_added_feature_select.options + w_data_select.value
            #Funcationality of load Filter class
            if w_event_filter_select.value:
                trimmed_options = [option for option in w_data_select.options if any(e in w_event_filter_select.value for e in self.data[option].event)]
                w_data_select.options = trimmed_options or []
        
        def data_feat_switch(event):
            w_data_select.value = []
            w_added_feature_select.value = []
            update_feature_info()
                 
        #adding buttons 
        load_feature_button = Button(name='Load', button_type='success')
        unload_feature_button = Button(name='Unload', button_type='danger')
        frequency_filter_button = Button(name='Frequency', button_type='primary')
        peak_value_filter_button = Button(name='Peak Value', button_type='primary')
        rising_time_filter_button = Button(name='Rising Time', button_type='primary')
        filter_button = Button(name='Filter', button_type='success')
        
        load_dendrogram_button = Button(name='Generate Dendrogram from Loaded Features', button_type='primary')
        load_cluster_button = Button(name='Load Cluster from Loaded Features', button_type='primary')
        
        load_feature_button.param.watch(load_feature, "clicks")
        unload_feature_button.param.watch(unload_feature, "clicks")
        load_dendrogram_button.param.watch(load_dendrogram, "clicks")
        load_cluster_button.param.watch(load_cluster,"clicks")
        filter_button.param.watch(load_filter,"clicks")
        
        # Register the callback with the value attribute of the feature selection widget               
        self.pn_data_features = pn.Tabs(('Initial Data',w_data_select),('Loaded Features',Column(w_added_feature_select,unload_feature_button)))
        self.pn_utility = Column(w_event_filter_select,self.w_time_filter,
                                 Column(frequency_filter_button,peak_value_filter_button,rising_time_filter_button),
                                 filter_button, width=300)
        self.pn_description = ('Description',Column(self.w_visualize, w_select_cell))
        self.pn_description_advanced = ('Description',Column(self.w_visualize, w_select_cell, w_distance_metric))
        self.pn_dendrogram = ('Dendrogram',Column(load_dendrogram_button, self.w_visualize_dendogram))
        self.pn_clustering = ('Cluster',Column(self.w_cluster_distance, load_cluster_button, self.w_visualize_cluster))
        
        self.pn_data_features.param.watch(data_feat_switch, 'active')
        self.main_panel = pn.Row(self.pn_data_features)

        w_data_select.param.watch(update_feature_info, 'value')
        w_added_feature_select.param.watch(update_feature_info, 'value')

    def _temp_comp_sub(self, usub=None, data=None):
        if usub is None:
            usub = self.strm_usub.usub
        
        signal = hv.Dataset(
                data.sel(unit_id=usub)
                .compute()
                .rename("Intensity (A. U.)")
                .dropna("frame", how="all")
            ).to(hv.Curve, "frame")
        return pn.panel(signal, width=400, height=200)
        
    
    def update_temp_comp_sub(self, usub=None):
        self.w_visualize.object = self._temp_comp_sub(usub, self.selection).object
    
    def show(self) -> Row:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolboxes.
        """
        return self.main_panel

    #def display(self):
    #    layout = self.show()
    #
    # layout.show()