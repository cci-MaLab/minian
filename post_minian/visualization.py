
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from post_minian.postprocessing import Feature
from panel.widgets import MultiSelect, StaticText, Select
from panel import Row, Column
from panel.layout import WidgetBox
import xarray as xr
import panel as pn

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
        self.features = {feature.name: feature for feature in features}
        self.widgets = self._create_widgets()
    
    def _create_widgets(self):
        # Implement multiselect for features, this will occupy the left side of the panel
        self.w_feature_select = MultiSelect(name='Feature Selection',
                options=[name for name in self.features.keys()])
        
        # Display information from selected feature

        # TODO: Plot the information from the selected feature
        self.w_description = StaticText(name="Description", value="")
        self.w_ranges = StaticText(name="Ranges", value="")
        self.w_events = StaticText(name="Events", value="")
        self.w_distance_metric = Select(name='Select', options=['Euclidean', 'Cosine', 'Manhattan'])

        # Set up the functionality of the widgets
        #@pn.depends(self.w_feature_select.param.value, watch=True)
        def update_description(selection):
            if selection:
                selection = selection[0]
                self.w_description.value = self.features[selection].description
                self.w_ranges.value = self.features[selection].ranges
                self.w_events.value = "".join(self.features[selection].event)
        
        self.w_feature_select.param.watch(update_description, 'value')

        # Set up the panels
        self.left_panel = self.w_feature_select
        self.right_panel_description = Column(self.w_description, self.w_ranges, self.w_events, self.w_distance_metric)

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
            self.right_panel_description            
        )
