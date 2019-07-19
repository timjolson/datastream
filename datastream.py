import logging
import os
import sys
import io
import string
import numpy as np
_ascii_fields = string.ascii_letters[23:26] + string.ascii_letters[0:23]


class DictArray():
    """Wrapper around np.array that can use keys for first axis.

    kwargs are passed to np.array after processing `data`
    DictArray(data=None, **kwargs)

    .keys() ~= dict().keys()
    .items() ~= dict().items()
    .values() ~= dict().values()
    .as_dict() returns dict version of .items()

    data can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
        `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
        *any `list` can be a sequence with __iter__

    np.array attributes are implicitly delegated to obj.points

    Examples:
        obj = DictArray([0,1])
        obj['x'] == obj[0] == obj[0,:] == [0]
        obj['x',0] == obj[0,0] == 0
        obj['y'] == obj[1] == obj[1,:] == [1]
        obj['y',0] == obj[1,0] == 1

        obj = DictArray([[0,2,4],[1,3,5])
        obj['x'] == obj[0] == obj[0,:] == [0,2,4]
        obj['x',1] == obj[0,1] == 1
        obj['y'] == obj[1] == obj[1,:] == [1,3,5]
        obj['y',1] == obj[1,1] == 3

    Written by Tim Olson - tim.lsn@gmail.com
    """
    def __init__(self, data=None, **kwargs):
        self._keys = ()
        self.points, self._keys = self._process_data(data, **kwargs)

    def keys(self):
        for k in self._keys:
            yield k

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def values(self):
        for k in self.keys():
            yield self[k]

    def as_dict(self):
        return dict(self.items())

    def _process_data(self, data, **kwargs):
        dt = data_type(data)
        if dt in (None, 'empty'):
            return np.array([], **kwargs), ()
        elif dt == 'listOfLists':
            sample = data[0]
            if hasattr(data, '_fields'):  # namedtuple
                names = data._fields
            elif hasattr(sample, '_fields'):  # namedtuple
                names = sample._fields
            else:
                names = self._keys or [_ascii_fields[i] for i in range(len(data))]
        elif dt == 'listOfDicts':
            sample = data[0]
            names = list(sample.keys())
            vv = []
            for k in sample.keys():  # curve names
                pp = []  # data for curve
                for p in data:  # each data point
                    pp.append(p[k])  # coordinate
                vv.append(pp)  # data for curve
            data = vv
        elif dt == 'listOfValues':
            if hasattr(data, '_fields'):  # namedtuple
                names = data._fields
            else:
                names = self._keys or [_ascii_fields[i] for i in range(len(data))]
            data = [[d] for d in data]
        elif dt == 'dictOfLists':
            names, data = list(data.keys()), list(data.values())
        elif dt == 'dictOfValues':
            names, data = list(data.keys()), [[d] for d in data.values()]
        elif dt == 'recarray':
            names = data.dtype['names']
        elif dt == 'ndarray':
            names = self._keys or [_ascii_fields[i] for i in range(len(data))]
        elif dt == 'DataStream':
            names = tuple(data._keys)
            data = data.points.copy()
        else:
            raise NotImplementedError(f"Cannot handle '{dt}' {data}")

        if self._keys and len(self._keys) != len(data): raise ValueError(
            f"Improper data count ({len(data)}) provided, need ({len(self._keys)})")

        data = np.array(data, **kwargs)

        if self._keys:  # we have column names
            sdata = np.ones_like(data) * np.nan  # sorted data

            for i, k in enumerate(self._keys):  # sort incoming data columns
                sdata[i, :] = data[names.index(k)][:]
            return sdata, self._keys
        else:  # set data column names
            self._keys = tuple(names)
            return data.copy(), tuple(names)

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            return getattr(self.points, attr)

    def _process_item_key(self, key):
        if isinstance(key, str):
            if key in self._keys:
                key = self._keys.index(key)
            else:  # force KeyError
                a = dict()[key]
        elif isinstance(key, tuple):
            if isinstance(key[0], str):  # handle string indexing -> column name
                if len(key) > 1:
                    key = self._keys.index(key[0]), *key[1:]
                else:
                    key = self._keys.index(key[0]),
        return key

    def __getitem__(self, key):
        key = self._process_item_key(key)
        return self.points.__getitem__(key)

    def __setitem__(self, key, value):
        key = self._process_item_key(key)
        return self.points.__setitem__(key, value)

    def __len__(self):
        if hasattr(self, 'points'):
            return self.points.__len__()
        else:
            return 0

    def __eq__(self, other):
        return other == self.points

    def __repr__(self):
        if self.points.size > 0:
            dd = {k: self.points[i] for i, k in enumerate(self._keys)}
        else:
            dd = {k: [] for i, k in enumerate(self._keys)}
        return f"{type(self).__name__}({dd})"


class DataStream(DictArray):
    """DictArray with data recording and convenience functions.

    .add_points(*data, **kwargs) where `data` matches a format
        supported by DictArray(), OR keywords can be used

    .set_record_file(file, format) set how/where to record data
        where `file` can be any of:
        str, io.IOBase, logging.Logger, logging.Handler, callable

    .start_recording()
    .stop_recording()

    Written by Tim Olson - tim.lsn@gmail.com
    """
    def __init__(self, data=None, record_to_file=None, pause=None, file_format='csv'):
        super().__init__(data, dtype=np.float64)
        self._file_inited = False
        self._recorder = lambda *x, **y: None
        self.file_format = file_format
        self._pause = pause if (pause is not None) else (record_to_file is None)
        self.set_record_file(record_to_file, format=file_format)

    def add_points(self, *data, **kwargs):
        """Add data points to DataStream, after some formatting and error checking.
        If applicable, records the new data via the set up recorder.

        :param data: data in format supported by DictArray()
        :param kwargs: keywords specifying data points
        :return:
        """
        if not data and not kwargs:
            raise ValueError("No point provided")
        elif not data and kwargs:
            data = kwargs
        elif len(data) == 1:
            data = data[0]

        data, names = self._process_data(data)
        oldlength = self.points.size
        if oldlength == 0:
            self.points = data.copy()
        else:
            self.points = np.concatenate([self.points, data], 1)
        if not self._pause and self._recorder:
            if not self._file_inited:
                self._init_file()
            for p in data.T:
                self._recorder(p)

    def start_recording(self):
        self._pause = False

    def stop_recording(self):
        self._pause = True

    def set_record_file(self, file, format='csv'):
        """Set a file/object/function used to record data points.

        :param file:
                str -> filepath -> .write()
                io.IOBase -> uses .write()
                logging.Logger -> uses Logger.info()
                logging.Handler -> uses Handler.handle(logging.getLogRecordFactory(`data`))
                callable -> calls, passing the new data point
        :param format: str: 'csv', 'dict', or 'list'
        :return:
        """
        # TODO: handle header loading

        if file is None:
            self.file = None
            self._recorder = lambda *x, **y: None
            return

        def fp(point):
            """string format point"""
            if isinstance(point, str):
                return point
            point = point.flatten()
            if self.file_format == 'csv':
                msg = ','.join(str(v) for v in point)
            elif self.file_format == 'dict':
                msg = str({k: v for k, v in zip(self._keys, point)})
            elif self.file_format == 'list':
                msg = str(list(point))
            else:
                raise NotImplementedError(f"File format '{self.file_format}' not recognized")
            return msg

        if isinstance(file, str):
            file = os.path.abspath(file)
            if not os.path.exists(file):  # file does not exist
                os.makedirs(os.path.dirname(file), exist_ok=True)  # make directory
                open(file, 'w').close()
            self._recorder = lambda msg: open(file, 'a').write(fp(msg) + os.linesep)
        elif isinstance(file, io.IOBase):
            self._recorder = lambda msg: file.write(fp(msg) + os.linesep)
        elif isinstance(file, logging.Logger):
            self._recorder = lambda msg: file.info(fp(msg))
        elif isinstance(file, logging.Handler):
            def make_record(msg):
                return logging.getLogRecordFactory()(
                    file._name, file._name, file.level, 'DataStream', 0, msg, (), sys.exc_info()
                )
            self._recorder = lambda msg: file.handle(make_record(fp(msg)))
        elif callable(file):
            self._recorder = file
        else:
            raise NotImplementedError(f"Don't know how to handle record_to_file={type(file)}")

        self.file_format = format
        self.file = file

    def _init_file(self):
        # TODO: handle header loading

        if self.file_format in ['csv', 'list']:
            header = ','.join((k if ',' not in k else f"\"{k}\"") for k in map(str, self._keys))
            self._recorder(header)
        elif self.file_format == 'dict':
            pass
        else:
            raise NotImplementedError(f"Unsupported format: '{self.file_format}'")
        self._file_inited = True

    def _record_point(self, point):
        if not self._pause and self._recorder:
            if not self._file_inited:
                self._init_file()
            self._recorder(point)


def data_type(obj):
    """Identify format of data object. Returns `str` or `None`"""
    # inspired by pyqtgraph.graphicsitems.PlotDataItem
    if obj is None:
        return None
    if hasattr(obj, '__len__') and len(obj) == 0:
        return 'empty'
    if isinstance(obj, DictArray):
        return 'DictArray'
    if isinstance(obj, dict):
        first = obj[list(obj.keys())[0]]
        if is_sequence(first):
            return 'dictOfLists'
        else:
            return 'dictOfValues'
    elif is_sequence(obj):
        first = obj[0]

        if (hasattr(obj, 'implements') and obj.implements('MetaArray')):
            return 'MetaArray'
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                if obj.dtype.names is None:
                    return 'listOfValues'
                else:
                    return 'recarray'
            # elif obj.ndim == 2 and obj.dtype.names is None and obj.shape[1] == 2:
            #     return 'Nx2array'
            elif obj.ndim == 2:
                return 'ndarray'
            else:
                raise Exception('array shape must be (N,) or (N,2); got %s instead' % str(obj.shape))
        elif isinstance(first, dict):
            return 'listOfDicts'
        elif is_sequence(first):
            return 'listOfLists'
        else:
            return 'listOfValues'


def is_sequence(obj):
    # inspired by pyqtgraph.graphicsitems.PlotDataItem
    return hasattr(obj, '__iter__') or isinstance(obj, np.ndarray) or (
                hasattr(obj, 'implements') and obj.implements('MetaArray'))


__all__ = ['DictArray', 'DataStream', 'is_sequence', 'data_type']
