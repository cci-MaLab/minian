import os
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_traces(explorer, neurons_to_plot=None, data_type='C', shift_amount=0.4):
    fig, ax = plt.subplots(figsize=(40,5))
    if data_type in ['C', 'S', 'E']:
        if neurons_to_plot == None:
            print("Please specify which, neurons to plot")
            return None
        shifts = [shift_amount * i for i in range(len(neurons_to_plot))]
        for shift, neuron in zip(shifts, neurons_to_plot):
            trace = explorer.data[data_type].sel(unit_id = neuron)
            trace /= np.max(trace)
    #         ax.autoscale()
            ax.plot(explorer.data['Time Stamp (ms)'],trace + shift,alpha=0.5)
            ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('RNFS')],0,5,color="green")
            ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('IALP')],0,5,color="blue")
            ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('ALP')],0,5,color="red",alpha=0.75)
    else:
        trace = explorer.data[data_type]
        max_trace = np.max(trace)
        ax.scatter(explorer.data['Time Stamp (ms)'],trace,alpha=0.5)
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('RNFS')],0,max_trace,color="green")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('IALP')],0,max_trace,color="blue")
        ax.vlines(explorer.data['Time Stamp (ms)'].loc[explorer.get_timestep('ALP')],0,max_trace,color="red",alpha=0.75)
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
