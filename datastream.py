import logging
import os
import sys
import io
import string
import numpy as np
import json
import csv
import ast
_ascii_fields = string.ascii_letters[23:26] + string.ascii_letters[0:23]


class DictArray():
    """Wrapper around np.array that can use keys to access first axis.

    DictArray(data=None, **kwargs) -> np.array(~data, **kwargs)
        data: can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
            `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
        kwargs: passed to np.array() after processing `data`
        NOTE: any `list` in above formats can be a sequence with __iter__

    Methods:
    .append(*data, **kwargs): add data to the array, where `data` fits format
            above, OR keywords can be used
    .keys() ~= dict().keys()
    .items() ~= dict().items()
    .values() ~= dict().values()
    .as_dict() returns dict version of .items()

    CAUTION: keys can be any valid dict key, except the types
        tuple/slice/list/int/np.ndarray that have no custom handling.

    All np.array attributes are implicitly delegated to DictArray().array

    Examples:
        obj = DictArray([[0,2,4],[1,3,5]])
        obj['x'] == obj[0] == obj[0,:] == [0,2,4]
        obj['x',1] == obj[0,1] == 2
        obj['y'] == obj[1] == obj[1,:] == [1,3,5]
        obj['y',1] == obj[1,1] == 3

        x, y = object(), object()
        obj = DictArray({x:[0], y:[1]})
        obj[x] == obj[0] == obj[0,:] == [0]
        obj[x,0] == obj[0,0] == 0
        obj[y] == obj[1] == obj[1,:] == [1]
        obj[y,0] == obj[1,0] == 1

    Written by Tim Olson - timjolson@user.noreplay.github.com
    """
    def __init__(self, data=None, **kwargs):
        self._keys = ()
        self.array, self._keys = process_data((), data, **kwargs)
        self._map_key_to_index = {k: i for i, k in enumerate(self._keys)}

    def append(self, *data, **kwargs):
        """Add data, after processing.

        :param data: data in format supported by DictArray()
        :param kwargs: keywords specifying data
        :return: np.ndarray: the processed, appended data
        """
        if not data and not kwargs:
            raise ValueError("No point provided")
        elif not data and kwargs:
            data = kwargs
        elif len(data) == 1:
            data = data[0]

        data, names = process_data(self._keys, data)
        oldlength = self.array.size
        if oldlength == 0:
            self.array = data.copy()
            self._keys = names
            self._map_key_to_index = {k: i for i, k in enumerate(names)}
        else:
            self.array = np.concatenate([self.array, data], 1)
        return data

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            return getattr(self.array, attr)

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

    def _process_item_key(self, key, axis=0):
        # TODO: add float handling
        if isinstance(key, (int, slice, np.ndarray)):
            pass
        elif isinstance(key, tuple):
            key = tuple(self._process_item_key(k, i) for i,k in enumerate(key))
        elif axis == 0:
            if isinstance(key, list):
                key = list(self._process_item_key(k, axis) for k in key)
            else:
                key = self._map_key_to_index[key]
        else:
            raise IndexError(f"index `{key}` not an option for axis {axis}")
        return key

    def __getitem__(self, key):
        key = self._process_item_key(key)
        return self.array[key]

    def __setitem__(self, key, value):
        key = self._process_item_key(key)
        return self.array.__setitem__(key, value)

    def __len__(self):
        if hasattr(self, 'array'):
            return self.array.__len__()
        else:
            return 0

    def __ge__(self, other):
        return self.array.__ge__(other)

    def __le__(self, other):
        return self.array.__le__(other)

    def __gt__(self, other):
        return self.array.__gt__(other)

    def __lt__(self, other):
        return self.array.__lt__(other)

    def __eq__(self, other):
        return self.array.__eq__(other)

    def __ne__(self, other):
        return self.array.__ne__(other)

    def __repr__(self):
        if self.array.size > 0:
            dd = {k: self.array[i] for i, k in enumerate(self._keys)}
        else:
            dd = {k: [] for i, k in enumerate(self._keys)}
        return f"{type(self).__name__}({dd})"


class DataStream(DictArray):
    """DictArray with data recording functions.

    .set_record_file(file, format)
        set a file/object/function used to record data points.
        `file`:
            str -> filepath -> .write(msg)
            io.IOBase -> uses .write(msg)
            logging.Logger -> uses Logger.info(msg)
            logging.Handler -> uses Handler.handle(logging.getLogRecordFactory(msg))
            callable -> calls, passing the new data point, unformatted
        `format`: str: 'csv', 'dict', or 'list'

    .start_recording()
    .stop_recording()

    Written by Tim Olson - timjolson@user.noreplay.github.com
    """
    def __init__(self, data=None, record_to_file=None, pause=None, file_format='csv', **kwargs):
        if isinstance(data, str):
            data, self.file_format = DataFileIO.parse_file(data)
        super().__init__(data, **kwargs)
        self._file_inited = False
        self._recorder = lambda *x, **y: None
        self._pause = pause if (pause is not None) else (record_to_file is None)
        self.set_record_file(record_to_file, format=file_format)

    def append(self, *data, **kwargs):
        data = super().append(*data, **kwargs)
        if not self._pause and self._recorder:
            if not self._file_inited:
                self._init_file()
            for p in data.T:
                self._recorder(p)
        return data

    def start_recording(self):
        self._pause = False

    def stop_recording(self):
        self._pause = True

    def set_record_file(self, file, format='csv'):
        """Set a file/object/function used to record data array.

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

def parse_data(data):
    names = None  # keys to return
    logger.info(f"parse_data {repr(data)}")
    dt = data_type(data)
    logger.info(f"dt={dt}")
    if dt is None:
        data = np.array([])
        return data, names
    elif dt == 'empty':
        # if isinstance(data, np.ndarray):
        #     if len(data.shape) > 0:
        #         return data, names
        return np.array([]), names
    elif dt == 'listOfLists':
        data_fields = data._fields if hasattr(data, '_fields') else None
        sample_fields = data[0]._fields if hasattr(data[0], '_fields') else None
        data = list(map(list, data))
        names = data_fields or sample_fields or None
        logger.info(f"listOfLists {repr(data)}")
        if data_fields:  # lists are in named tuple
            data = np.array(data).T
            logger.info(f"listOfLists {repr(data)}")
        elif sample_fields:  # lists are named tuples
            logger.info(f"listOfLists SAMPLE")
        else:
            data = np.array(data)
            logger.info(f"listOfLists {repr(data)}")
        logger.info(f"listOfLists -> {repr(data)}")
        return data, names
    elif dt == 'listOfDicts':
        sample = data[0]
        names = list(sample.keys())
        vvv = []
        for d in data:
            vv = []
            for i, (k, v) in enumerate(d.items()):  # order, name, values
                vv.append(v)
            vvv.append(vv)
        data = vvv
        return np.array(data), tuple(names)
    elif dt == 'listOfValues':
        if hasattr(data, '_fields'):  # namedtuple
            names = data._fields
        # data = np.array([[v for v in data]])
        data = np.array([[*data]])
        return data, names
    elif dt == 'dictOfLists':
        names, data = tuple(data.keys()), list(map(lambda *a: list(a), *data.values()))  # transpose data
        data = np.array(data)
        logger.info(f'dictOfLists {names}, {repr(data)}')
        if data.size == 0:
            logger.info('make to size')
            data.resize((0, len(names)))
        return data, names
    elif dt == 'dictOfValues':
        names, data = tuple(data.keys()), [list(data.values())]
        logger.info(f"dictOfValues {repr(data)}")
        data = np.array(data)
        logger.info(f"dictOfValues {repr(data)}")
        return data, names
    elif dt == 'recarray':
        names = data.dtype.names
        logger.info(f"recarray {repr(data)}")
        if data.size == 0:
            data = np.ndarray((0, len(names)))
        else:
            data = np.array(list(map(list, data)))  # get list version to remove dtype // np.void
        logger.info(f"recarray {repr(data)}")
        return data, names
    elif dt == 'ndarray':
        names = data.dtype.names
        data = np.array(list(map(list, data)))  # get list version to remove dtype // np.void
        return data, names
    elif dt == 'DictArray':
        names = data._keys
        data = data.array
        return data, names
    else:
        raise NotImplementedError(f"Cannot handle '{dt}' {data}")

    if keys and len(keys) != len(data): raise ValueError(
        f"Improper data count ({len(data)}) provided, need ({len(keys)})")

    data = np.array(data, **kwargs)

    if keys:  # we have column names
        sdata = np.ones_like(data) * np.nan  # sorted data

        for i, k in enumerate(keys):  # sort incoming data columns
            sdata[i, :] = data[names.index(k)][:]
        return sdata, keys
    else:  # set data column names
        return data.copy(), tuple(names)


def data_type(obj):
    """Identify format of data object. Returns `str` or `None`"""
    # inspired by pyqtgraph.graphicsitems.PlotDataItem
    logger.info(f"data_type {type(obj), repr(obj)}")
    if obj is None:
        return None
    if isinstance(obj, DictArray):
        return 'DictArray'
    if hasattr(obj, '__len__'):
        if isinstance(obj, np.ndarray):
            pass
        elif len(obj) == 0:
            return 'empty'
    if isinstance(obj, dict):
        first = obj[list(obj.keys())[0]]
        if is_sequence(first):
            return 'dictOfLists'
        else:
            return 'dictOfValues'
    if is_sequence(obj):
        # if (hasattr(obj, 'implements') and obj.implements('MetaArray')):
        #     return 'MetaArray'
        # elif isinstance(obj, np.ndarray):
        if isinstance(obj, np.ndarray):
            if obj.ndim <= 1:
                if obj.dtype.names is None:
                    return 'listOfValues'
                elif obj.dtype.names:
                    return 'recarray'
                # else:
                #     return 'empty'
            # elif obj.ndim == 2 and obj.dtype.names is None and obj.shape[1] == 2:
            #     return 'Nx2array'
            elif obj.ndim == 2:
                if obj.dtype.names is None:
                    return 'ndarray'
                else:
                    return 'recarray'
            else:
                raise Exception('array shape must be (N points, N keys); got %s instead' % str(obj.shape))

        # try:
        first = obj[0]
        # except IndexError:
        #     return 'empty'

        if isinstance(first, dict):
            return 'listOfDicts'
        elif is_sequence(first):
            if isinstance(first, np.ndarray) and first.ndim == 0:
                return 'listOfValues'
            else:
                return 'listOfLists'
        else:
            return 'listOfValues'


def is_sequence(obj):
    # inspired by pyqtgraph.graphicsitems.PlotDataItem
    return hasattr(obj, '__iter__') or isinstance(obj, np.ndarray) or (
                hasattr(obj, 'implements') and obj.implements('MetaArray'))


class DataFileIO():
    @staticmethod
    def parse_file(filename):
        logger.info(f"parse_file `{filename}`")
        data, ff = DataFileIO.detect_format(filename)

        if data is not None:
            logger.info((data, ff))
            return data, ff

        if ff['file'] == 'empty':
            logger.info((data, ff))
            return data, ff
        elif ff['file'] == 'numpy':
            data = DataFileIO.do_numpy_csv(ff, filename)
        elif ff['file'] is None:
            try:
                data = json.load(open(filename))
            except json.JSONDecodeError as e:
                logger.info(f"Doing numpy {ff}")
                data = DataFileIO.do_numpy_csv(ff, filename)
        elif ff['file'] == 'multiLine':
            logger.info(f"Doing multiLine {ff}")
            data = DataFileIO.do_multiline_literal(ff, filename)
        else:
            raise NotImplementedError(ff)

        logger.info((data, ff))
        return data, ff

    @staticmethod
    def readnlines(file, n=0):
        if n == -1:
            return file.readlines()
        if n == 0:
            return file.read()
        rv = []
        for i in range(n):
            line = file.readline()
            if line:
                rv.append(line)
        return rv

    @staticmethod
    def detect_format(filename):
        logger.info(f"detect_format {filename}")
        file = open(filename)
        ff = {'data': None, 'file': None, 'header': None, 'dialect': None}

        try:
            samplelines = DataFileIO.readnlines(file, 4)
        except UnicodeDecodeError as e:
            ff['file'] = 'numpy'
            logger.info(ff)
            return None, ff

        n_samplelines = len(samplelines)

        if n_samplelines in [0, 1]:
            if n_samplelines == 0 or samplelines[0].strip(' \t\n\r') == '':
                ff['file'] = 'empty'
                logger.info(ff)
                return [], ff

        logger.info('here')
        if samplelines[0][0] in "[({":
            try:
                sample = ast.literal_eval(samplelines[0])
                logger.info('here')
            except SyntaxError:
                logger.info('here')
                if samplelines[0][1] in "[({":
                    sample = ast.literal_eval(samplelines[0][1:])
                    logger.info(f'sample {sample}')
                else:
                    file.seek(0)
                    sample = ast.literal_eval(file.read())
                    ff['file'] = 'multiLine'
                    ff['data'] = data_type(sample)
                    return sample, ff
            if isinstance(sample, dict):
                ff['data']='dict'
                itemsample = list(sample.values())[0]
                if is_sequence(itemsample):
                    ff['data']+='OfLists'
                elif n_samplelines == 1:
                    ff['data']+='OfValues'
                else:
                    ff['data']='listOfDicts'

                if n_samplelines == 1:
                    ff['file'] = 'oneLine'
                else:
                    ff['file'] = 'multiLine'

            elif is_sequence(sample):
                ff['data']='list'
                itemsample = sample[0]
                if isinstance(itemsample, dict):
                    ff['data']+='OfDicts'
                elif is_sequence(itemsample):
                    ff['data']+='OfLists'
                else:
                    ff['data']+='OfValues'

                if n_samplelines == 1:
                    ff['file'] = 'oneLine'
                else:
                    ff['file'] = 'multiLine'
            else:
                raise TypeError(type(sample), sample)
        else:
            samplestr = ''.join(samplelines)
            logger.info(f'samplestr {samplestr}')
            try:
                ff['dialect'] = csv.Sniffer().sniff(samplestr)
                logger.info(f"dialect {ff['dialect']}")
            except csv.Error:
                logger.info(f'csv Error')
                if n_samplelines > 1:
                    logger.info(f"MORE THAN ONE LINE")
                    sample_without_header = ''.join(samplelines[1:])
                    try:
                        ff['dialect'] = csv.Sniffer().sniff(sample_without_header)
                    except csv.Error:
                        pass
                    else:
                        ff['header'] = samplelines[0]
            logger.info('CHECK HEADER')
            if not ff['header']:
                try:
                    hasheader = csv.Sniffer().has_header(samplestr)
                except csv.Error:
                    logger.info(f'DOES NOT HAVE HEADER')
                    pass
                else:
                    logger.info(f'HAS HEADER {hasheader}')
                    ff['header'] = samplelines[0] if hasheader else None

        logger.info(ff)

        if ff['file'] == 'oneLine':
            logger.info((sample, ff))
            return sample, ff
        logger.info((None, ff))
        return None, ff

    @staticmethod
    def do_multiline_literal(ff, filename):
        logger.info(filename)
        file = open(filename)

        if ff['data'] in ['listOfLists', 'dictOfLists']:
            data = ast.literal_eval(file.read())
        elif ff['data'] in ['listOfDicts', 'listOfValues']:
            data = [ast.literal_eval(l) for l in file.readlines()]
        elif ff['data'] == 'unknown':
            logger.info(f"Doing unknown {ff}")
            data = ast.literal_eval(file.read())
            ff['data'] = data_type(data)
            logger.info(f"Doing unknown {ff}, {repr(data)}")
        else:
            raise NotImplementedError(ff)

        return data

    @staticmethod
    def do_numpy_csv(ff, filename):
        try:
            data = np.load(filename)
            ff['data'] = 'save'
            ff['file'] = 'numpy'
        except ValueError:
            kw = {'fname':filename}
            if ff['dialect']:
                kw.update(delimiter=ff['dialect'].delimiter)
            if ff['header']:
                kw.update(names=True)
            data = np.genfromtxt(**kw)
            ff['data'] = 'text'
            ff['file'] = 'csv'

        if data.dtype.names is not None:
            ff['header'] = data.dtype.names
            if ff['data'] != 'text':
                ff['data'] += '-structuredarray'

        return data
parse_file = DataFileIO.parse_file

__all__ = ['DictArray', 'DataStream']
