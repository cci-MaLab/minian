import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union
from panel.widgets import MultiSelect

def plot_multiple_traces(explorer, neurons_to_plot=None, data_type='C', shift_amount=0.4, figure_ax = None):
    if figure_ax == None:
        fig, ax = plt.subplots(figsize=(40,5))
    else:
        ax = figure_ax
    if data_type in ['C', 'S', 'E']:
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
        features: Union[xr.DataArray, List[xr.DataArray]]
    ):
        self.features = features
        self.widgets = self._create_widgets()
    
    def _widgets(self):
        # Implement multiselect for features
        multi_select = MultiSelect(name='Feature Selection',
                options=[feature.name for feature in self.features])
        

class VArrayViewer:
    """
    Interactive visualization for movie data arrays.

    Hint
    ----
    .. figure:: img/vaviewer.png
        :width: 500px
        :align: left

    The visualization contains following panels from top to bottom:

    Play Toolbar
        A toolbar that controls playback of the video. Additionally, when the
        button "Update Mask" is clicked, the coordinates of the box drawn in
        *Current Frame* panel will be used to update the `mask` attribute of the
        `VArrayViewer` instance, which can be later used to subset the data. If
        multiple arrays are visualized and `layout` is `False`, then drop-down
        lists corresponding to each metadata dimensions will show up so the user
        can select which array to visualize.
    Current Frame
        Images of the current frame. If multiple movie array are passed in,
        multiple frames will be labeled and shown. To the side of each frame
        there is a histogram of intensity values. The "Box Select" tool can be
        used on the histogram to limit the range of intensity used for
        color-mapping. Additionally, the "Box Edit Tool" is available for use on
        the frame image, where you can hold "Shift" and draw a box, whose
        coordinates can be used to update the `mask` attribute of the
        `VarrayViewer` instance (remember to click "Update Mask" after drawing).
    Summary
        Summary statistics of each frame across time. Only shown if `summary` is
        not empty. The red vertical line indicate current frame.

    Attributes
    ----------
    mask : dict
        Instance attribute that can be retrieved and used to subset data later.
        Keys are `tuple` with values corresponding to each `meta_dims` and
        uniquely identify each input array. If `meta_dims` is empty then keys
        will be empty `tuple` as well. Values are `dict` mapping dimension names
        (of the arrays) to subsetting slices. The slices are in the plotting
        coorandinates and can be directly passed to `xr.DataArray.sel` method to
        subset data.
    """

    def __init__(
        self,
        varr: Union[xr.DataArray, List[xr.DataArray], xr.Dataset],
        framerate=30,
        summary=["mean"],
        meta_dims: List[str] = None,
        datashading=True,
        layout=False,
    ):
        """
        Parameters
        ----------
        varr : Union[xr.DataArray, List[xr.DataArray], xr.Dataset]
            Input array, list of arrays, or dataset to be visualized. Each array
            should contain dimensions "height", "width" and "frame". If a
            dataset, then the dimensions specified in `meta_dims` will be used
            as metadata dimensions that can uniquely identify each array. If a
            list, then a dimension "data_var" will be constructed and used as
            metadata dimension, and the `.name` attribute of each array will be
            used to identify each array.
        framerate : int, optional
            The framerate of playback when using the toolbar. By default `30`.
        summary : list, optional
            List of summary statistics to plot. The statistics should be one of
            `{"mean", "max", "min", "diff"}`. By default `["mean"]`.
        meta_dims : List[str], optional
            List of dimension names that can uniquely identify each input array
            in `varr`. Only used if `varr` is a `xr.Dataset`. By default `None`.
        datashading : bool, optional
            Whether to use datashading on the summary statistics. By default
            `True`.
        layout : bool, optional
            Whether to visualize all arrays together as layout. If `False` then
            only one array will be visualized and user can switch array using
            drop-down lists below the *Play Toolbar*. By default `False`.

        Raises
        ------
        NotImplementedError
            if `varr` is not a `xr.DataArray`, a `xr.Dataset` or a list of `xr.DataArray`
        """
        if isinstance(varr, list):
            for iv, v in enumerate(varr):
                varr[iv] = v.assign_coords(data_var=v.name)
            self.ds = xr.concat(varr, dim="data_var")
            meta_dims = ["data_var"]
        elif isinstance(varr, xr.DataArray):
            self.ds = varr.to_dataset()
            self.varr_copy = varr.copy()
            self.varr = varr
            self.can_change = True
        elif isinstance(varr, xr.Dataset):
            self.ds = varr
        else:
            raise NotImplementedError(
                "video array of type {} not supported".format(type(varr))
            )
        if not self.can_change:
            print("Warning: Frame fixing disabled. Make sure to load in varr as dataArray.")
        try:
            self.meta_dicts = OrderedDict(
                [(d, list(self.ds.coords[d].values)) for d in meta_dims]
            )
            self.cur_metas = OrderedDict(
                [(d, v[0]) for d, v in self.meta_dicts.items()]
            )
        except TypeError:
            self.meta_dicts = dict()
            self.cur_metas = dict()
        self._datashade = datashading
        self._layout = layout
        self.framerate = framerate
        self._f = self.ds.coords["frame"].values
        self._h = self.ds.sizes["height"]
        self._w = self.ds.sizes["width"]
        self._first = 0
        self._last = 0
        self.mask = dict()
        CStream = Stream.define(
            "CStream",
            f=param.Integer(
                default=int(self._f.min()), bounds=(self._f.min(), self._f.max())
            ),
        )
        RStream = Stream.define(
            "RStream",
            r=param.Range(
                default=(int(self._f.min()), int(self._f.min())),
                bounds=(int(self._f.min()), int(self._f.max())),
            ),
        )
        self.strm_f = CStream()
        self.strm_r = RStream()
        self.str_box = BoxEdit()
        self.widgets = self._widgets()
        self._summary_types = summary.copy() if type(summary) is list else None
        if type(summary) is list:
            summ_all = {
                "mean": self.ds.mean(["height", "width"]),
                "max": self.ds.max(["height", "width"]),
                "min": self.ds.min(["height", "width"]),
                "diff": self.ds.diff("frame").mean(["height", "width"]),
            }
            try:
                summ = {k: summ_all[k] for k in summary}
            except KeyError:
                print("{} Not understood for specifying summary".format(summary))
            if summ:
                print("computing summary")
                sum_list = []
                for k, v in summ.items():
                    sum_list.append(v.compute().assign_coords(sum_var=k))
                summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        if layout:
            self.ds_sub = self.ds
            self.sum_sub = self.summary
        else:
            self.ds_sub = self.ds.sel(**self.cur_metas)
            try:
                self.sum_sub = self.summary.sel(**self.cur_metas)
            except AttributeError:
                self.sum_sub = self.summary
        self.pnplot = pn.panel(self.get_hvobj())

    def update_ds(self):
        if type(self._summary_types) is list:
            # Rewrote this slightly for efficiency. We assume no key errors at this point.
            summ = {}
            if "mean" in self._summary_types:
                summ["mean"] = self.ds.mean(["height", "width"])
            if "max" in self._summary_types:
                summ["max"] = self.ds.max(["height", "width"])
            if "min" in self._summary_types:
                summ["min"] = self.ds.min(["height", "width"])
            if "diff" in self._summary_types:
                summ["diff"] = self.ds.diff("frame").mean(["height", "width"])

            if summ:
                sum_list = []
                for k, v in summ.items():
                    sum_list.append(v.compute().assign_coords(sum_var=k))
                summary = xr.concat(sum_list, dim="sum_var")
        self.summary = summary
        if self._layout:
            self.ds_sub = self.ds
            self.sum_sub = self.summary
        else:
            self.ds_sub = self.ds.sel(**self.cur_metas)
            try:
                self.sum_sub = self.summary.sel(**self.cur_metas)
            except AttributeError:
                self.sum_sub = self.summary

        self.pnplot.object = self.get_hvobj()
        self.update_status.value = "Updated Visualization (Done!)"

    def get_hvobj(self):
        def get_im_ovly(meta):
            def img(f, ds):
                return hv.Image(ds.sel(frame=f).compute(), kdims=["width", "height"])

            try:
                curds = self.ds_sub.sel(**meta).rename("_".join(meta.values()))
            except ValueError:
                curds = self.ds_sub
            fim = fct.partial(img, ds=curds)
            im = hv.DynamicMap(fim, streams=[self.strm_f]).opts(
                frame_width=500, aspect=self._w / self._h, cmap="Viridis"
            )
            self.xyrange = RangeXY(source=im).rename(x_range="w", y_range="h")
            if not self._layout:
                hv_box = hv.Polygons([]).opts(
                    style={"fill_alpha": 0.3, "line_color": "white"}
                )
                self.str_box = BoxEdit(source=hv_box)
                im_ovly = im * hv_box
            else:
                im_ovly = im

            def hist(f, w, h, ds):
                if w and h:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    ).select(height=h, width=w)
                else:
                    cur_im = hv.Image(
                        ds.sel(frame=f).compute(), kdims=["width", "height"]
                    )
                return hv.operation.histogram(cur_im, num_bins=50).opts(
                    xlabel="fluorescence", ylabel="freq"
                )

            fhist = fct.partial(hist, ds=curds)
            his = hv.DynamicMap(fhist, streams=[self.strm_f, self.xyrange]).opts(
                frame_height=int(500 * self._h / self._w), width=150, cmap="Viridis"
            )
            im_ovly = (im_ovly << his).map(lambda p: p.opts(style=dict(cmap="Viridis")))
            return im_ovly

        if self._layout and self.meta_dicts:
            im_dict = OrderedDict()
            for meta in itt.product(*list(self.meta_dicts.values())):
                mdict = {k: v for k, v in zip(list(self.meta_dicts.keys()), meta)}
                im_dict[meta] = get_im_ovly(mdict)
            ims = hv.NdLayout(im_dict, kdims=list(self.meta_dicts.keys()))
        else:
            ims = get_im_ovly(self.cur_metas)
        if self.summary is not None:
            hvsum = (
                hv.Dataset(self.sum_sub)
                .to(hv.Curve, kdims=["frame"])
                .overlay("sum_var")
            )
            if self._datashade:
                hvsum = datashade_ndcurve(hvsum, kdim="sum_var")
            try:
                hvsum = hvsum.layout(list(self.meta_dicts.keys()))
            except:
                pass
            

            hvsum *= hv.DynamicMap(lambda r: self._return_hvArea(r[0], r[1]),
                streams=[self.strm_r]).opts(style=dict(alpha=0.1, color="red")
            )
            vl = hv.DynamicMap(lambda f: hv.VLine(f), streams=[self.strm_f]).opts(
                style=dict(color="red")
            )
            summ = (hvsum * vl).map(
                lambda p: p.opts(frame_width=500, aspect=3), [hv.RGB, hv.Curve]
            )
            hvobj = (ims + summ).cols(1)
        else:
            hvobj = ims
        return hvobj
    
    def _return_hvArea(self, _first, _last):
        _max = list(self.sum_sub.max().values())[0].item() # I HATE XARRAYS
        if _first >= _last:
            _first, _last = 0, 0
        return hv.Area((np.arange(_first, _last), np.ones(_last-_first) * _max),
                kdims=["frame"])


    def show(self) -> pn.layout.Column:
        """
        Return visualizations that can be directly displayed.

        Returns
        -------
        pn.layout.Column
            Resulting visualizations containing both plots and toolbars.
        """
        return pn.layout.Column(self.widgets, self.pnplot)

    def _widgets(self):
        w_play = pnwgt.Player(
            length=len(self._f), interval=int(1000 / self.framerate), value=0, width=650, height=90
        )

        if self.can_change:
            frame_value = pnwgt.StaticText(name="Current Frame", value="0")
            frame_ranges = pnwgt.StaticText(name="Frame Ranges", value="0 - 0")
            self.update_status = pnwgt.StaticText(name="Frame Status", value="No Changes")

        def play(f):
            if not f.old == f.new:
                _f = int(self._f[f.new])
                self.strm_f.event(f=_f)
                frame_value.value = str(_f)
        
        def _update_first(click):
            self._first = self.strm_f.f
            frame_ranges.value = f"{self._first} - {self._last}"
            self.strm_r.event(r=(self._first, self._last))
        
        def _update_last(click):
            self._last = self.strm_f.f
            frame_ranges.value = f"{self._first} - {self._last}"
            self.strm_r.event(r=(self._first, self._last))

        w_play.param.watch(play, "value")
        w_box = pnwgt.Button(
            name="Update Mask", button_type="primary", width=100, height=30
        )
        w_box.param.watch(self._update_box, "clicks")
        if self.can_change:
            w_first = pnwgt.Button(
                name="First Good Frame", button_type="primary", width=100, height=30
            )
            w_first.param.watch(_update_first, "clicks")
            w_last = pnwgt.Button(
                name="Last Good Frame", button_type="primary", width=100, height=30
            )
            w_last.param.watch(_update_last, "clicks")
            w_interpolate = pnwgt.Button(
                name="Interpolate", button_type="primary", width=100, height=30
            )
            w_interpolate.param.watch(self._fill_interpolate, "clicks")
            w_fill_left = pnwgt.Button(
                name="Fill From Left", button_type="primary", width=100, height=30
            )
            w_fill_left.param.watch(self._fill_left, "clicks")
            w_fill_right = pnwgt.Button(
                name="Fill From Right", button_type="primary", width=100, height=30
            )
            w_fill_right.param.watch(self._fill_right, "clicks")
            w_reset = pnwgt.Button(
                name="Reset", button_type="primary", width=100, height=30
            )
            w_reset.param.watch(self._reset, "clicks")
        if not self._layout:
            wgt_meta = {
                d: pnwgt.Select(name=d, options=v, height=45, width=120)
                for d, v in self.meta_dicts.items()
            }

            def make_update_func(meta_name):
                def _update(x):
                    self.cur_metas[meta_name] = x.new
                    self._update_subs()

                return _update

            for d, wgt in wgt_meta.items():
                cur_update = make_update_func(d)
                wgt.param.watch(cur_update, "value")
            
            if self.can_change:
                wgts = pn.layout.WidgetBox(pn.Row(w_box, w_first, w_last, frame_ranges, frame_value),
                                        w_play, pn.Row(w_interpolate, w_fill_left, w_fill_right, self.update_status),
                                        w_reset, *list(wgt_meta.values()))
            else:
                wgts = pn.layout.WidgetBox(w_box,
                                        w_play, *list(wgt_meta.values()))
                
        else:
            if self.can_change:
                wgts = pn.layout.WidgetBox(pn.Row(w_box, w_first, w_last, frame_ranges, frame_value),
                                       w_play, pn.Row(w_interpolate, w_fill_left, w_fill_right, self.update_status),
                                       w_reset)
            else:
                wgts = pn.layout.WidgetBox(w_box, w_play)
        return wgts
    def _reset(self, click):
        self.varr_copy = self.varr.copy()
        self.ds = self.varr_copy.to_dataset()
        self.update_ds()

    def _update_subs(self):
        self.ds_sub = self.ds.sel(**self.cur_metas)
        if self.sum_sub is not None:
            self.sum_sub = self.summary.sel(**self.cur_metas)
        self.pnplot.objects[0].object = self.get_hvobj()

    def _update_box(self, click):
        box = self.str_box.data
        self.mask.update(
            {
                tuple(self.cur_metas.values()): {
                    "height": slice(box["y0"][0], box["y1"][0]),
                    "width": slice(box["x0"][0], box["x1"][0]),
                }
            }
        )
