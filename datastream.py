import logging
import os
import sys
import io
import string
import numpy as np
import json
import csv
import ast
import utils
from collections import OrderedDict
logger = logging.getLogger(__name__)


class DictArray():
    """Wrapper around np.ndarray that allows key indexing (like dict of arrays) and
            indexing/slicing in getitem/setitem.

    NOTE: Because the keys are used as indices, they must be HASHABLE and CANNOT be:
            iterable, tuple, list, int, slice, numpy.array

    DictArray(data=None, keys=None)
        data: can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
            `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
            NOTE: any `list` in above formats can be a sequence with __iter__
        keys: dict or iterable; dict will rename data by its own keys, iterable will rename in order.
            Keys will be automatically generated when needed from DictArray.DictArray_default_keys;
            When keys are renamed, the new name is used for indexing.

    Methods:
        .append(*data, **kwargs): add data to the array, where `data` fits format
                above, OR keywords can be used
        .extend(*data, **kwargs): like append, but adds columns/keys of data (len must match already stored data)
        .keys() ~= dict.keys()
        .items() ~= dict.items()
        .values() ~= dict.values()
        .dict() returns dict version of .items()

    All np.array attributes are implicitly delegated to obj.array

    Key/Index Usage:
        When `key` is used as an index for axis 0, the effective indexing is `[:, key]`
            therefore other indices cannot also be used:
                obj['x'] -> ok  ;  obj[1,'x'] -> ok  ;  obj['x',1] -> TypeError
        When slicing, keys can be used for start and stop, but not step. They also
            convert the slice into INCLUDING the stop index:
            obj = DictArray([[1,2,3],[4,5,6]])
            obj['x':'y'] == obj['x':2] == obj[0:'y'] -> [[1,2],[4,5]]
            obj['x':'z'] == obj['x':3] == obj[0:'z']-> [[1,2,3],[4,5,6]]
            obj[0:2:'x'] -> KeyError  ;  obj['x':'z':'y'] -> KeyError
        When slicing, keys can also be used to reverse ordering:
            obj['z':'y'] == obj[2:1:-1] == obj[2:'y':-1] -> [[3,2],[6,5]]

    Key setting
        DictArray([[0,1],[2,3]], keys=('a', 'b'))  # keys specified by iterable, assigned in order
        DictArray([[0,1],[2,3]], keys={'y':'second', 'x':'first'})  # by dict, renames the auto-generated keys
        DictArray({'a':[1,1],'b':[2,2]}, keys={'a':'ones', 'b':'twos'})  # by dict, renames incoming keys
        DictArray([[0,1],[0,1]]).rename('x', 'zeros')  # a renamed key is indexed by its new name (`zeros` here)

    Examples:
        obj = DictArray([[0,1],[2,3]])  # keys auto-generated
        obj['x'] == obj[:, 0] == obj[:,'x'] == [0,2]
        obj['y'][1] == obj[1,1] == obj[1,'y'] == 3

        x, y = object(), object()
        obj = DictArray({x:[0], y:[1]})  # keys specified in data
        obj[y] == obj[:,1] == obj[:,y] == [1]
        obj[y][0] == obj[0][1] == obj[0,1] == obj[0,y] == 1

    Written by Tim Olson - timjolson@user.noreplay.github.com
    """
    DictArray_default_keys = string.ascii_letters[23:26] + string.ascii_letters[0:23]
    DictArray_default_keys = list(''.join([DictArray_default_keys, DictArray_default_keys.upper()]))

    def __init__(self, data=None, keys=None):
        logger.debug(f"data={data}, keys={keys}")
        self.default_keys = self.DictArray_default_keys.copy()
        self._keys = ()
        self._rename_dict = rnd = OrderedDict()
        self.array = np.array([])

        if data is not None:
            self.extend(data)

        if keys is None:
            return

        # keys are specified
        extra_keys = OrderedDict()
        if not isinstance(keys, dict):  # iterable, use in order
            if len(keys) < len(self._keys):
                raise ValueError(f"At least ({len(self._keys)}) keys are needed, provided ({len(keys)})")
            new = 0
            for i, k in enumerate(keys):  # handle renaming, spare keys
                if i < len(self._keys):
                    rnd[self._keys[i]] = k
                else:
                    extra_keys[self.default_keys[new]] = k
                    new += 1
        else:
            for k, v in keys.items():  # handle renaming, spare keys
                if k not in rnd.keys():
                    extra_keys[k] = v
                else:
                    self.rename(k, v)

        if extra_keys:  # extend data for spare keys
            logger.info(f'have extra keys {extra_keys}')
            self.extend(OrderedDict([(v,np.full(self.array.shape[0], np.nan)) for v in extra_keys.values()]))

        self._keys = tuple(rnd.values())  # update _keys for indexing, iterating
        logger.info(f"after init keys {self._keys}, {repr(self.array)}, {self.default_keys}")

    def rename(self, input_key, new_access_key):
        """Rename an incoming data key to index it by another key

        :param input_key: key for incoming data
        :param new_access_key: new key to use for indexing
        :return:
        """
        logger.info(f"rename {input_key}, {new_access_key}")
        if new_access_key in self._rename_dict.values():
            raise ValueError(f"Key `{new_access_key}` already assigned")
        old_access_key = self._rename_dict.get(input_key, None)
        logger.info(f"rename old_access_key = {old_access_key}")

        if old_access_key is None:
            self.extend({input_key:np.full([self.array.shape[0]], np.nan)})
            self.rename(input_key, new_access_key)
            return

        if old_access_key in self.DictArray_default_keys:
            logger.info(f're-add {old_access_key}')
            self.default_keys.append(old_access_key)

        self._rename_dict[input_key] = new_access_key
        for k in self._rename_dict.values():
            try: self.default_keys.remove(k)
            except ValueError: pass
        self._keys = tuple(self._rename_dict.values())

    def extend(self, *data, **data_kw):
        """Process and add data points as columns/new keys

        :param data: data in format supported by DictArray()
        :param data_kw: keywords specifying data
        :return:
        """
        logger.info(f'extend {data}, {data_kw}')
        if not data and data_kw:
            data = data_kw
        elif len(data) == 1:
            data = data[0]
        array, parse_keys = parse_data(data)

        if parse_keys == 0:  # new data is useless
            return
        elif isinstance(parse_keys, int):  # auto-generate keys
            parse_keys = self.default_keys[:parse_keys]

        for k in parse_keys:  # update rename dict with new keys
            if k in self._keys:
                raise ValueError(f"Key {k} already assigned")
            self._rename_dict[k] = k
        for k in self._rename_dict.values():  # update default_keys list
            try: self.default_keys.remove(k)
            except ValueError: pass
        self._keys = tuple(self._rename_dict.values())  # update self._keys

        if self.array.ndim == 2:  # current data has keys
            if array.shape[0] != self.array.shape[0]:  # check shape of new data
                raise ValueError(f"New data shape ({array.shape[0]}) does not match existing ({self.array.shape[0]})")
            self.array = np.column_stack([self.array, array])
        else:
            self.array = array

    def append(self, *data, **data_kw):
        """Process and add data points

        :param data: data in format supported by DictArray()
        :param data_kw: keywords specifying data
        :return: np.ndarray: the processed, appended data
        """
        logger.info(f"append {data}, {data_kw}")
        if not data and data_kw:
            data = data_kw
        elif len(data) == 1:
            data = data[0]
        data, parse_keys = _parse_data_apply_keys(self, data)
        logger.info(f"append parse {repr(data)}, {parse_keys}")

        s = self.array.shape
        if self.array.size == 0 and self.array.ndim <= 1:  # no current data, no keys yet
            logger.info(f"append no data, no keys")
            if isinstance(parse_keys, int):  # data does not have keys, has key count
                parse_keys = self.default_keys[:parse_keys]

            logger.info(f"append updating keys")
            self.array.resize(data.shape)
            self.array[:] = data[:]
            for k in parse_keys:
                try: self.default_keys.remove(k)
                except ValueError: pass
            self._keys = tuple(parse_keys)
            self._rename_dict.update((k,k) for k in parse_keys)
            return data

        # have keys already set
        if parse_keys == self._keys:  # keys are same, in same order
            logger.info(f"append same keys")
            new_data = data  # set as sorted
            keys = parse_keys
        else:  # not same keys or different order
            logger.info(f"append different keys")
            keys = list(parse_keys)  # so we can use .index
            logger.info(f'shapes {self.array.shape, data.shape}')
            new_data = np.full((data.shape[0],self.array.shape[1]), np.nan)  # sorted data
            for i, k in enumerate(self._keys):  # sort incoming data columns
                logger.info(f"append sorting {i,k}")
                try:                            # to match self._keys
                    new_index = keys.index(k)
                    logger.info(f"append new data has key {k} index {new_index}")
                except ValueError:
                    new_data[:, i] = np.full(data.shape[0], np.nan)
                    logger.info(f"append new data does not have key {k}, {new_data}")
                else:
                    new_data[:, i] = data[:, new_index]
                    logger.info(f"append new data {k}, {new_data}")

        # actually append data
        logger.info(f'Appending {new_data} to {self.array}')
        self.array.resize(s[0] + new_data.shape[0], s[1])
        self.array[s[0]:, :] = new_data[:]

        extra_data = OrderedDict()
        for i, k in enumerate(keys):
            if k not in self._keys:  # extra key / extend data
                extra_data[k] = np.concatenate(
                    [np.full((self.array.shape[0]-data.shape[0]),np.nan), data[:,i]]
                )
        if extra_data:
            self.extend(extra_data)
        return new_data

    # <editor-fold> convenience funcs
    def keys(self):
        """Generator of keys ~= dict.keys()"""
        for k in self._keys:
            yield k

    def items(self):
        """Generator of key,value ~= dict.items()"""
        for k in self.keys():
            yield k, self[k]

    def values(self):
        """Generator of values ~= dict.values()"""
        for k in self.keys():
            yield self[k]

    def dict(self):
        """Get as dict of arrays"""
        return dict(self.items())
    # </editor-fold>

    # <editor-fold> getitem & setitem
    def __getitem__(self, key):
        # TODO: add float handling
        logger.info(f"getitem `{key}`")
        if utils.contains_non_index(key):
            if isinstance(key, tuple):  # ['x'] OK // NOT OK ['x', 0]
                if utils.contains_non_index(key[0]) and len(key) > 1:
                    raise TypeError("Keys cannot be used on axis 0 while indexing other axes")
                key = tuple(utils.process_getitem_key(self, k, i) for i, k in enumerate(key))
            elif isinstance(key, slice):  # ['x':'z']  [0:'y']  ['x':2:1] OK // NOT OK [~:~:'x']
                key = slice(None), utils.do_keyed_slice(self, key)
            else:
                key = slice(None), utils.process_getitem_key(self, key)
        return self.array[key]

    def __setitem__(self, key, value):
        # TODO: add float handling
        logger.info(f"getitem `{key}`")
        if utils.contains_non_index(key):
            if isinstance(key, tuple):  # ['x'] OK // NOT OK ['x', 0]
                if utils.contains_non_index(key[0]) and len(key) > 1:
                    raise TypeError("Keys cannot be used on axis 0 while indexing other axes")
                key = tuple(utils.process_getitem_key(self, k, i) for i, k in enumerate(key))
            elif isinstance(key, slice):  # ['x':'z']  [0:'y']  ['x':2:1] OK // NOT OK [~:~:'x']
                key = slice(None), utils.do_keyed_slice(self, key)
            else:
                key = slice(None), utils.process_getitem_key(self, key)
        return self.array.__setitem__(key, value)
    # </editor-fold>

    # <editor-fold> array delegation
    def __len__(self):
        return self.array.__len__()

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

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            return getattr(self.array, attr)

    def __repr__(self):
        if self.array.size > 0:
            dd = {k: self.array[:, i] for i, k in enumerate(self._keys)}
        else:
            dd = {k: [] for i, k in enumerate(self._keys)}
        return f"{type(self).__name__}({dd})"
    # </editor-fold>


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


def parse_data(data):
    names = ()  # keys to return
    logger.debug(f"parse_data `{repr(data)}`")
    if data is None:
        return np.array([]), ()
    dt = data_type(data)
    logger.debug(f"dt={dt}")
    if dt == 'empty':
        logger.info(f"DOING empty {repr(data)}")
        return np.array([]), ()
    elif dt == 'listOfLists':
        data_fields = data._fields if hasattr(data, '_fields') else None
        sample_fields = data[0]._fields if hasattr(data[0], '_fields') else None
        data = list(map(list, data))
        names = data_fields or sample_fields or ()
        logger.debug(f"listOfLists {repr(data)}")
        if data_fields:  # lists are in named tuple
            data = np.array(data).T
            logger.debug(f"listOfLists {repr(data)}")
        else:
            data = np.array(data)
        logger.debug(f"listOfLists -> {repr(data)}")
        return data, names or ((data.shape[1] or names) if data.ndim == 2 else names)
    elif dt == 'listOfDicts':
        sample = data[0]
        names = list(sample.keys())
        vvv = []
        for d in data:
            vv = []
            for i, (k, v) in enumerate(d.items()):  # order, name, values
                vv.append(v)
            vvv.append(vv)
        data = np.array(vvv)
        return data, tuple(names) or (data.shape[1] if data.ndim == 2 else names)
    elif dt == 'listOfValues':
        if hasattr(data, '_fields'):  # namedtuple
            names = data._fields
        data = np.array([[*data]])
        return data, names or (data.shape[1] if data.ndim == 2 else 0)
    elif dt == 'dictOfLists':
        names, data = tuple(data.keys()), list(map(lambda *a: list(a), *data.values()))  # transpose data
        data = np.array(data)
        logger.debug(f'dictOfLists {names}, {repr(data)}')
        if data.size == 0:
            logger.debug('make to size')
            data.resize((0, len(names)))
        return data, names or (data.shape[1] if data.ndim == 2 else names)
    elif dt == 'dictOfValues':
        names, data = tuple(data.keys()), [list(data.values())]
        logger.debug(f"dictOfValues {repr(data)}")
        data = np.array(data)
        logger.debug(f"dictOfValues {repr(data)}")
        return data, names or (data.shape[1] if data.ndim == 2 else names)
    elif dt == 'recarray':
        names = data.dtype.names
        logger.debug(f"recarray {repr(data)}")
        if data.size == 0:
            data = np.ndarray((0, len(names)))
        else:
            data = np.asanyarray(data.tolist())  # get list version to remove dtype // np.void
        logger.debug(f"recarray {repr(data)}")
        return data, names or (data.shape[1] if data.ndim == 2 else names)
    elif dt == 'ndarray':
        return data.copy(), data.shape[1] or names
    elif dt == 'DictArray':
        return data.array.copy(), data._keys or names
    else:
        raise NotImplementedError(f"Cannot handle '{dt}' {data}")


def data_type(obj):
    """Identify format of data object. Returns `str` or `None`"""
    # inspired by pyqtgraph.graphicsitems.PlotDataItem
    logger.debug(f"data_type {type(obj), repr(obj)}")
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
        if utils.is_sequence(first):
            return 'dictOfLists'
        else:
            return 'dictOfValues'
    if utils.is_sequence(obj):
        # if (hasattr(obj, 'implements') and obj.implements('MetaArray')):
        #     return 'MetaArray'
        if isinstance(obj, np.ndarray):
            if obj.ndim <= 1:
                if obj.dtype.names is None:
                    return 'listOfValues'
                elif obj.dtype.names:
                    return 'recarray'
            # elif obj.ndim == 2 and obj.dtype.names is None and obj.shape[1] == 2:
            #     return 'Nx2array'
            elif obj.ndim == 2:
                if obj.dtype.names is None:
                    return 'ndarray'
                else:
                    return 'recarray'
            else:
                raise Exception('array shape must be (N points, N keys); got %s instead' % str(obj.shape))

        first = obj[0]

        if isinstance(first, dict):
            return 'listOfDicts'
        elif utils.is_sequence(first):
            if isinstance(first, np.ndarray) and first.ndim == 0:
                return 'listOfValues'
            else:
                return 'listOfLists'
        else:
            return 'listOfValues'
    raise NotImplementedError(f"Unknown data_type, {repr(obj)}")


def _parse_data_apply_keys(self, *data, **data_kw):
    logger.info(f'_parse_data_apply_keys {data}, {data_kw}')
    if len(data) == 1:
        data = data[0]

    array, parse_keys = parse_data(data)

    if not isinstance(parse_keys, int) and self._rename_dict:  # take keys and rename them
        logger.info(f'_parse_data_apply_keys renaming')
        parse_keys = tuple(map(lambda k: self._rename_dict.get(k, k), parse_keys))
    elif self._rename_dict:
        logger.info(f'_parse_data_apply_keys looking up')
        parse_keys = []
        len_keys = len(self._keys)
        new = 0
        for i in range(array.shape[1]):
            if i < len_keys:
                result = self._keys[i]
            else:
                result = self.default_keys[new]
                new += 1
            parse_keys.append(result)
    elif parse_keys == 0:  # make automatic names
        logger.info(f'_parse_data_apply_keys generating {array.shape[1]} from {self.default_keys}')
        parse_keys = self.default_keys[:array.shape[1]]
    logger.info(f'_parse_data_apply_keys {repr(array)}, {parse_keys}')

    return array, parse_keys


class DataFileIO():
    @staticmethod
    def parse_string(string, newline=None):
        file = io.StringIO(string, newline)
        return DataFileIO.parse_file(file)

    @staticmethod
    def parse_file(file):
        logger.debug(f"parse_file `{file}`")
        if isinstance(file, str):
            file = open(file, 'r')
        data, ff = DataFileIO.detect_format(file)

        if data is not None:
            logger.debug((data, ff))
            return data, ff

        if ff['file'] == 'empty':
            logger.debug((data, ff))
            return data, ff
        elif ff['file'] == 'numpy':
            data = DataFileIO.do_numpy_csv(ff, file)
        elif ff['file'] is None:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                logger.debug(f"Doing numpy {ff}")
                data = DataFileIO.do_numpy_csv(ff, file)
        elif ff['file'] == 'multiLine':
            logger.debug(f"Doing multiLine {ff}")
            data = DataFileIO.do_multiline_literal(ff, file)
        else:
            raise NotImplementedError(ff)

        logger.debug((data, ff))
        return data, ff

    @staticmethod
    def readnlines(file, n=0):
        if n == -1:
            rv = file.readlines()
            file.seek(0)
            return rv
        if n == 0:
            rv = file.read()
            file.seek(0)
            return rv
        rv = []
        for i in range(n):
            line = file.readline()
            if line:
                rv.append(line)
        file.seek(0)
        return rv

    @staticmethod
    def detect_format(file):
        logger.debug(f"detect_format {file}")
        ff = {'data': None, 'file': None, 'header': None, 'dialect': None}

        try:
            samplelines = DataFileIO.readnlines(file, 3)
        except UnicodeDecodeError as e:
            ff['file'] = 'numpy'
            logger.debug(ff)
            return None, ff

        n_samplelines = len(samplelines)

        if n_samplelines in [0, 1]:
            if n_samplelines == 0 or samplelines[0].strip(' \t\n\r') == '':
                ff['file'] = 'empty'
                logger.debug(ff)
                return [], ff

        logger.debug(samplelines)
        if samplelines[0][0] in "[({":
            try:
                sample = ast.literal_eval(samplelines[0])
                logger.debug('here')
            except SyntaxError:
                logger.debug('here')
                if samplelines[0][1] in "[({":
                    sample = ast.literal_eval(samplelines[0][1:])
                    logger.debug(f'sample {sample}')
                else:
                    # file.seek(0)
                    sample = ast.literal_eval(file.read())
                    ff['file'] = 'multiLine'
                    ff['data'] = data_type(sample)
                    return sample, ff
            if isinstance(sample, dict):
                ff['data'] = 'dict'
                itemsample = list(sample.values())[0]
                if utils.is_sequence(itemsample):
                    ff['data'] += 'OfLists'
                elif n_samplelines == 1:
                    ff['data'] += 'OfValues'
                else:
                    ff['data'] = 'listOfDicts'

                if n_samplelines == 1:
                    ff['file'] = 'oneLine'
                else:
                    ff['file'] = 'multiLine'

            elif utils.is_sequence(sample):
                ff['data'] = 'list'
                itemsample = sample[0]
                if isinstance(itemsample, dict):
                    ff['data'] += 'OfDicts'
                elif utils.is_sequence(itemsample):
                    ff['data'] += 'OfLists'
                else:
                    ff['data'] += 'OfValues'

                if n_samplelines == 1:
                    ff['file'] = 'oneLine'
                else:
                    ff['file'] = 'multiLine'
            else:
                raise TypeError(type(sample), sample)
        else:
            samplestr = ''.join(samplelines)
            logger.debug(f'samplestr {samplestr}')
            try:
                ff['dialect'] = csv.Sniffer().sniff(samplestr)
                logger.debug(f"dialect {ff['dialect']}")
            except csv.Error:
                logger.debug(f'csv Error')
                if n_samplelines > 1:
                    logger.debug(f"MORE THAN ONE LINE")
                    sample_without_header = ''.join(samplelines[1:])
                    try:
                        ff['dialect'] = csv.Sniffer().sniff(sample_without_header)
                    except csv.Error:
                        pass
                    else:
                        ff['header'] = samplelines[0]
            logger.debug('CHECK HEADER')
            if not ff['header']:
                try:
                    hasheader = csv.Sniffer().has_header(samplestr)
                except csv.Error:
                    logger.debug(f'DOES NOT HAVE HEADER')
                    pass
                else:
                    logger.debug(f'HAS HEADER {hasheader}')
                    ff['header'] = samplelines[0] if hasheader else None

        if ff['file'] == 'oneLine':
            logger.debug((sample, ff))
            return sample, ff
        logger.debug((None, ff))
        return None, ff

    @staticmethod
    def do_multiline_literal(ff, file):
        logger.debug(file)

        if ff['data'] in ['listOfLists', 'dictOfLists']:
            data = ast.literal_eval(file.read())
        elif ff['data'] in ['listOfDicts', 'listOfValues']:
            data = [ast.literal_eval(l) for l in file.readlines()]
        elif ff['data'] == 'unknown':
            logger.debug(f"Doing unknown {ff}")
            data = ast.literal_eval(file.read())
            ff['data'] = data_type(data)
            logger.debug(f"Doing unknown {ff}, {repr(data)}")
        else:
            raise NotImplementedError(ff)

        return data

    @staticmethod
    def do_numpy_csv(ff, file):
        logger.debug(f"do_numpy_csv {ff}, {file}")
        file.seek(0)
        try:
            data = np.load(file)
            logger.debug(f"do_numpy_csv loaded")
            ff['data'] = 'save'
            ff['file'] = 'numpy'
        except UnicodeDecodeError as e:
            logger.debug(f"do_numpy_csv did not load {repr(e)}")
            file.seek(0)
            newfile = io.BytesIO(file.buffer.read())
            data = DataFileIO.do_numpy_csv(ff, newfile)
            logger.debug(f"data from BytesIO {repr(data)}")
            return data
        except (TypeError, ValueError) as e:
            logger.debug(f"do_numpy_csv did not load {repr(e)}")
            logger.debug(f"do_numpy_csv did not load, trying genfromtxt")
            file.seek(0)
            kw = {'fname': file}
            if ff['dialect']:
                kw.update(delimiter=ff['dialect'].delimiter)
            if ff['header']:
                kw.update(names=True)
            data = np.genfromtxt(**kw)
            ff['data'] = 'text'
            ff['file'] = 'csv'

        logger.debug(f"do_numpy_csv {data}")

        if data.dtype.names is not None:
            ff['header'] = data.dtype.names
            if ff['data'] != 'text':
                ff['data'] += '-structuredarray'
        logger.debug(f"do_numpy_csv {ff}")
        return data
parse_file = DataFileIO.parse_file

__all__ = ['DictArray', 'DataStream']
