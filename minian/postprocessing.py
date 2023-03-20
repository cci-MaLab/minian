from typing import Optional
import xarray as xr
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import cm
import matplotlib.pyplot as plt



class CellClustering:
    """
    Cell clustering class. This class is used to cluster cells based on their
    temporal activity, using FFT and agglomerative clustering.
    """

    def __init__(
        self,
        minian: Optional[xr.Dataset] = None,
        signals: Optional[xr.Dataset] = None,
        A: Optional[xr.DataArray] = None
    ):
    
        self.signals = signals if signals is not None else minian['C']
        self.A = A if A is not None else minian['A']
        self.units = self.signals['unit_id'].values
        self.psd_list = []

        for unit in self.units:
            self.compute_psd(unit) # compute psd for each unit
        
        # Compute agglomerative clustering
        self.linkage_data = linkage(self.psd_list, method='ward', metric='cosine')
        print("hello")

    def compute_psd(self, unit: int):
        val = self.signals.sel(unit_id=unit).values
        f, psd = welch(val,
               fs=1./30,
               window='hann',
               nperseg=256,
               detrend='constant') 
        self.psd_list.append(psd)
    
    def visualize_dendrogram(self):
        return dendrogram(self.linkage_data)
        
    
    def visualize_clusters(self, distance):
        self.cluster_indices = fcluster(self.linkage_data, distance, criterion='distance')
        viridis = cm.get_cmap('viridis', self.cluster_indices.max())
        
        image_shape = self.A.sel(unit_id=self.units[0]).values.shape
        final_image = np.zeros((image_shape[0], image_shape[1], 3))

        for idx, cluster in enumerate(self.cluster_indices):
            final_image += np.stack((self.A.sel(unit_id=self.units[idx]).values,)*3, axis=-1) * viridis(cluster)[:3]
        
        return plt.imshow(final_image)
