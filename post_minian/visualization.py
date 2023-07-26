
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from post_minian.postprocessing import Feature, CellClustering
from panel.widgets import MultiSelect, StaticText, Select, Button
from panel import Row, Column
from panel.layout import WidgetBox
import xarray as xr
import holoviews as hv
import panel as pn
from holoviews.streams import Stream
import param

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
        features: Union[Feature, List[Feature]],
        A: xr.DataArray,
    ):
        self.features = features
        self._all_cells = None

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
        w_feature_select = MultiSelect(name='Feature Selection',
                options=[feature.name for feature in self.features])
        
        #adding features that are TRUE in MultiSelect
        w_added_feature_select = MultiSelect(name='Loaded Features')
        
        # Display information from selected feature
        w_select_cell = Select(name='Select Cell', options=[])
        self.w_visualize = pn.panel(hv.Curve([]).opts(xaxis=None,yaxis=None,xlabel=None,ylabel=None), width=400, height=200)
        
        self.w_visualize_dendogram = pn.panel(hv.Curve([]).opts(xaxis=None,yaxis=None,xlabel=None,ylabel=None), width=400, height=200)
        self.w_visualize_cluster = pn.panel(hv.Curve([]).opts(xaxis=None,yaxis=None,xlabel=None,ylabel=None), width=400, height=200)
        
        w_description = StaticText(name="Description", value="")
        w_ranges = StaticText(name="Ranges", value="")
        w_events = StaticText(name="Events", value="")
        w_distance_metric = Select(name='Select', options=['Euclidean', 'Cosine', 'Manhattan'])

        def update_usub(usub):
            to_int = int(usub.new.split(" ")[1])
            self.strm_usub.event(usub=to_int)

        w_select_cell.param.watch(update_usub, "value")
        
        def update_feature_info(event):
            selected_features = event.new
            if selected_features:
                selected_feature_name = selected_features[0]
                selected_feature = next(
                    (feature for feature in self.features if feature.name == selected_feature_name),
                    None
                )
                
                w_description.value = selected_feature.description
                w_ranges.value = selected_feature.ranges
                w_events.value = selected_feature.event

                # Visualization stuff
                self.data = selected_feature.values
                self._all_cells = self.data.isel(frame=0).dropna("unit_id").coords["unit_id"].values
                w_select_cell.options = [f"Cell {u}" for u in self._all_cells]
                self.update_temp_comp_sub(self._all_cells[0])
                
                #self.update_cell_clustering()
        
        def load_feature(clicks=None):
            if w_feature_select.value:
                w_added_feature_select.options = w_added_feature_select.options + w_feature_select.value
        
        def unload_feature(clicks=None):
            if w_added_feature_select.value:
                trimmed_options = [option for option in w_added_feature_select.options if option not in w_added_feature_select.value]
                w_added_feature_select.options = trimmed_options
                
        #adding buttons 
        load_feature_button = Button(name='Load', button_type='success')
        unload_feature_button = Button(name='Unload', button_type='danger')
        
        load_feature_button.param.watch(load_feature, "clicks")
        unload_feature_button.param.watch(unload_feature, "clicks")
        
        
        # Register the callback with the value attribute of the feature selection widget               
        self.left_panel = pn.Tabs(('Inital Features',w_feature_select),('Loaded Features',w_added_feature_select))
        self.middle_panel = Column(load_feature_button,unload_feature_button)
        self.right_panel_description = pn.Tabs(
            ('Description',Column(self.w_visualize, w_select_cell, w_description, w_ranges, w_events, w_distance_metric)),
            ('Dendogram',self.w_visualize_dendogram),
            ('Cluster',self.w_visualize_cluster)
        )
        
        w_feature_select.param.watch(update_feature_info, 'value')
        w_added_feature_select.param.watch(update_feature_info, 'value')
        load_feature()
        
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
        self.w_visualize.object = self._temp_comp_sub(usub, self.data).object
    
    '''
    #I was not sure what should be the input of this functions. 
    #Section or A values in CellClustering 
        
    def update_cluster(self,cluster):
        self.w_visualize_cluster = cluster 
    
    def update_dendogram(self,dendogram):
        self.w_visualize_dendogram = dendogram
        
    def update_cell_clustering(self,Section,A,distance=None):
        if not Section and not A:
            temp_cell_clustering = CellClustering(Section,A)
            self.update_cluster(temp_cell_clustering.visualize_clusters(distance))
            self.update_dendogram(temp_cell_clustering.visualize_dendrogram())
    ''' 
    
    def show(self) -> Row:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolboxes.
        """
        return Row(
            self.left_panel,
            self.middle_panel,
            self.right_panel_description
        )