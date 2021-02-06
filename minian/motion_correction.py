import functools as fct
import itertools as itt

import cv2
import dask.array as darr
import numpy as np
import xarray as xr

from .utilities import xrconcat_recursive


def estimate_shifts(varr, max_sh, dim="frame", npart=3, local=False):
    """
    Estimate the frame shifts

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        max_sh (integer): maximum shift
        dim (str, optional): name of the z-dimension, defaults to "frame".
        npart (integer, optional): [the number of partitions of the divide-and-conquer algorithm]. Defaults to 3.
        local (boolean, optional): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value]. Defaults to False.

    Returns:
        xarray.DataArray: the estimated shifts
    """
    varr = varr.transpose(..., dim, "height", "width")
    loop_dims = list(set(varr.dims) - set(["height", "width", dim]))
    if loop_dims:
        loop_labs = [varr.coords[d].values for d in loop_dims]
        res_dict = dict()
        for lab in itt.product(*loop_labs):
            va = varr.sel({loop_dims[i]: lab[i] for i in range(len(loop_dims))})
            vmax, sh = est_sh_part(va.data, max_sh, npart, local)
            sh = xr.DataArray(
                sh,
                dims=[dim, "variable"],
                coords={dim: va.coords[dim].values, "variable": ["height", "width"],},
            )
            res_dict[lab] = sh.assign_coords(**{k: v for k, v in zip(loop_dims, lab)})
        sh = xrconcat_recursive(res_dict, loop_dims)
    else:
        vmax, sh = est_sh_part(varr.data, max_sh, npart, local)
        sh = xr.DataArray(
            sh,
            dims=[dim, "variable"],
            coords={dim: varr.coords[dim].values, "variable": ["height", "width"],},
        )
    return sh


def est_sh_part(varr, max_sh, npart, local, min_temp_fm=30):
    """
    Estimate the shift per frame

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        max_sh (integer): maximum shift
        npart (integer): [the number of partitions of the divide-and-conquer algorithm].
        local (boolean): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value].

    Returns:
        xarray.DataArray: the max shift per frame
        xarray.DataArray: the shift per frame
    """
    if varr.shape[0] <= 1:
        return varr.squeeze(), np.array([[0, 0]])
    elif varr.shape[0] > npart * min_temp_fm:
        match_func = darr.gufunc(
            match_temp,
            signature="(h,w),(h,w),(),()->(s)",
            output_dtypes=float,
            output_sizes={"s": 2},
        )
        shift_func = darr.gufunc(
            shift_perframe, signature="(h,w),(s)->(h,w)", output_dtypes=float,
        )
        if varr.shape[0] > npart ** 2 * min_temp_fm:
            part_func = est_sh_part
        else:
            part_func = darr.gufunc(
                est_sh_part,
                signature="(f,h,w),(),(),(),()->(h,w),(f,s)",
                output_dtypes=[float, float],
                output_sizes={"s": 2},
                allow_rechunk=True,
            )
    else:
        part_func = est_sh_part
        match_func = match_temp
        shift_func = shift_perframe
        npart = varr.shape[0]
    idx_spt = np.array_split(np.arange(varr.shape[0]), npart)
    fm_ls, sh_ls = [], []
    for idx in idx_spt:
        if len(idx) > 0:
            fm, sh = part_func(varr[idx, :, :], max_sh, npart, local, min_temp_fm)
            fm_ls.append(fm)
            sh_ls.append(sh)
    mid = int(len(sh_ls) / 2)
    sh_add_ls = [np.array([0, 0])] * len(sh_ls)
    for i, fm in enumerate(fm_ls):
        if i < mid:
            temp = fm_ls[i + 1]
            sh_idx = np.arange(i + 1)
        elif i > mid:
            temp = fm_ls[i - 1]
            sh_idx = np.arange(i, len(sh_ls))
        else:
            continue
        sh_add = match_func(fm, temp, max_sh, local)
        for j in sh_idx:
            sh_ls[j] = sh_ls[j] + sh_add.reshape((1, -1))
            sh_add_ls[j] = sh_add_ls[j] + sh_add
    for i, (fm, sh) in enumerate(zip(fm_ls, sh_add_ls)):
        fm_ls[i] = darr.nan_to_num(shift_func(fm, sh))
    sh_ret = darr.concatenate(sh_ls)
    fm_ret = darr.stack(fm_ls)
    return fm_ret.max(axis=0), sh_ret


def match_temp(src, dst, max_sh, local, subpixel=False):
    """
    Match template.
    For more information on template matching: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

    Args:
        src (array): source frame
        dst (array): template
        max_sh (integer): maximum shift
        local (boolean): [in case where there are multiple local maximum of the cross-correlogram, setting this to `True` will constraint the shift to be the one that’s closest to zero shift. i.e. this assumes the shifts are always small and local regardless of the correlation value].
        subpixel (boolean, optional): [whether to estimate shifts to sub-pixel level using polynomial fitting of cross-correlogram]. Defaults to False.

    Returns:
        [array]: array (x,y) of the shift (match)
    """
    src = np.pad(src, max_sh)
    cor = cv2.matchTemplate(
        src.astype(np.float32), dst.astype(np.float32), cv2.TM_CCOEFF_NORMED
    )
    if not len(np.unique(cor)) > 1:
        return np.array([0, 0])
    cent = np.floor(np.array(cor.shape) / 2)
    if local:
        cor_ma = cv2.dilate(cor, np.ones((3, 3)))
        maxs = np.array(np.nonzero(cor_ma == cor))
        dev = ((maxs - cent[:, np.newaxis]) ** 2).sum(axis=0)
        imax = maxs[:, np.argmin(dev)]
    else:
        imax = np.unravel_index(np.argmax(cor), cor.shape)
    if subpixel:
        x0 = np.arange(max(imax[0] - 5, 0), min(imax[0] + 6, cor.shape[0]))
        x1 = np.arange(max(imax[1] - 5, 0), min(imax[1] + 6, cor.shape[1]))
        y0 = cor[x0, imax[1]]
        y1 = cor[imax[0], x1]
        p0 = np.polyfit(x0, y0, 2)
        p1 = np.polyfit(x1, y1, 2)
        imax = np.array([-0.5 * p0[1] / p0[0], -0.5 * p1[1] / p1[0]])
        # m0 = np.roots(np.polyder(p0)) # for higher order polynomial fit
        # m1 = np.roots(np.polyder(p1))
        # m0 = m0[np.argmin(np.abs(m0 - imax[0]))]
        # m1 = m1[np.argmin(np.abs(m1 - imax[1]))]
        # imax = np.array([m0, m1])
    sh = cent - imax
    return sh


def apply_shifts(varr, shifts, fill=np.nan):
    """
    Apply the shifts to the input frames

    Args:
        varr (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the videos with dimensions: frame, height and width.
        shifts (xarray.DataArray): xarray.DataArray a labeled 3-d array representation of the shifts with dimensions: frame, height and width.

    Returns:
        (xarray.DataArray): xarray.DataArray of the shifted input frames (varr)
    """
    sh_dim = shifts.coords["variable"].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ["variable"]],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def shift_perframe(fm, sh, fill=np.nan):
    """
    Determine the shift per frame

    Args:
        fm (array): array with the pixels of the frame
        sh (array): (x,y) shift

    Returns:
        array: frame
    """
    if np.isnan(fm).all():
        return fm
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    index = [slice(None) for _ in range(fm.ndim)]
    for ish, s in enumerate(sh):
        index = [slice(None) for _ in range(fm.ndim)]
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = fill
        elif s == 0:
            continue
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = fill
    return fm
