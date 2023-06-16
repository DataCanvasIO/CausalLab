import base64
import glob
import os.path as path
import threading
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
from scipy import interpolate


def load_data(data_path, *, reset_index=False, reader_mapping=None, **kwargs):
    """
    load dataframe from data_path
    """

    if reader_mapping is None:
        reader_mapping = {
            'csv': partial(pd.read_csv, low_memory=False),
            'txt': partial(pd.read_csv, low_memory=False),
            'parquet': pd.read_parquet,
            'par': pd.read_parquet,
            'json': pd.read_json,
            'pkl': pd.read_pickle,
            'pickle': pd.read_pickle,
        }

    def get_file_format(file_path):
        return path.splitext(file_path)[-1].lstrip('.')

    def get_file_format_by_glob(data_pattern):
        for f in glob.glob(data_pattern, recursive=True):
            fmt_ = get_file_format(f)
            if fmt_ in reader_mapping.keys():
                return fmt_
        return None

    if glob.has_magic(data_path):
        fmt = get_file_format_by_glob(data_path)
    elif not path.exists(data_path):
        raise ValueError(f'Not found path {data_path}')
    elif path.isdir(data_path):
        path_pattern = f'{data_path}*' if data_path.endswith(path.sep) else f'{data_path}{path.sep}*'
        fmt = get_file_format_by_glob(path_pattern)
    else:
        fmt = path.splitext(data_path)[-1].lstrip('.')

    if fmt not in reader_mapping.keys():
        raise ValueError(f'Not supported data format{fmt}')
    fn = reader_mapping[fmt]
    df = fn(data_path, **kwargs)

    if reset_index:
        df.reset_index(drop=True, inplace=True)

    return df


def load_b64data(data, filename, *, reset_index=False, reader_mapping=None, **kwargs):
    """
    load dataframe from data_path
    """

    if reader_mapping is None:
        reader_mapping = {
            'csv': partial(pd.read_csv, low_memory=False),
            'txt': partial(pd.read_csv, low_memory=False),
            'parquet': pd.read_parquet,
            'par': pd.read_parquet,
            'json': pd.read_json,
            'pkl': pd.read_pickle,
            'pickle': pd.read_pickle,
        }

    fmt = path.splitext(filename)[-1].lstrip('.')

    if fmt not in reader_mapping.keys():
        raise ValueError(f'Not supported data format{fmt}')
    fn = reader_mapping[fmt]

    data = base64.b64decode(data)
    buf = BytesIO(data)
    df = fn(buf, **kwargs)

    if reset_index:
        df.reset_index(drop=True, inplace=True)

    return df


def smooth_line(xs, ys):
    # see: https://github.com/kawache/Python-B-spline-examples

    # tck, u = interpolate.splprep([xs, ys], k=3, s=0)
    # u = np.linspace(0, 1, num=len(xs) * 3, endpoint=True)
    # out = interpolate.splev(u, tck)

    n = len(xs)

    t = np.linspace(0, 1, n - 2, endpoint=True)
    t = np.append([0, 0, 0], t)
    t = np.append(t, [1, 1, 1])

    tck = [t, [xs, ys], 3]
    u = np.linspace(0, 1, (max(n * 2, 30)), endpoint=True)
    out = interpolate.splev(u, tck)
    return out


def smooth_line_bak(xs, ys):
    xs_orig, ys_orig = xs.copy(), ys.copy()
    xs = np.array(xs)
    ys = np.array(ys)

    flip_xy = False
    flip_ud = False
    if np.all(xs[1:] > xs[:-1]):
        pass
    elif np.all(xs[1:] < xs[:-1]):
        xs, ys = np.flipud(xs), np.flipud(ys)
        flip_ud = True
    elif np.all(ys[1:] > ys[:-1]):
        xs, ys = ys, xs
        flip_xy = True
    elif np.all(ys[1:] < ys[:-1]):
        xs, ys = np.flipud(ys), np.flipud(xs)
        flip_xy = True
        flip_ud = True
    else:
        # not found monotonic direction, skip smooth
        print('xs=', xs_orig)
        print('ys=', ys_orig)
        print('skip smooth')
        return xs_orig, ys_orig

    step_min = (xs[1:] - xs[:-1]).min()
    num = min(int((xs[-1] - xs[0]) / step_min * 1), 100)
    values_x = np.linspace(start=xs[0], stop=xs[-1], num=num, endpoint=True)
    spline = interpolate.PchipInterpolator(xs, ys)
    values_y = spline(values_x)

    if flip_ud:
        values_x, values_y = np.flipud(values_x), np.flipud(values_y)
    if flip_xy:
        values_x, values_y = values_y, values_x

    return values_x, values_y


def _proc_stub(target, args=None, kwargs=None, on_success=None, on_error=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    try:
        r = target(*args, **kwargs)
        if on_success is not None:
            on_success(r)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if on_error is not None:
            on_error(e)


def trun(target, args=None, kwargs=None, on_success=None, on_error=None):
    """
    Run target function on thread
    :return: threading.Thread
    """
    t = threading.Thread(
        target=partial(_proc_stub, target,
                       args=args,
                       kwargs=kwargs,
                       on_success=on_success,
                       on_error=on_error)
    )
    t.start()

    return t
