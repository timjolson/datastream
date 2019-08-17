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

_ascii_fields = string.ascii_letters[23:26] + string.ascii_letters[0:23]
_ascii_fields = list(_ascii_fields.join([_ascii_fields, _ascii_fields.upper()]))
logger = logging.getLogger(__name__)


class DictArray():
    """Wrapper around np.ndarray that can use keys to access first axis.

    DictArray(data=None, **dtype) -> np.array(~data, **dtype)
        data: can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
            `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
        dtype: passed to np.array() after processing `data`
        NOTE: any `list` in above formats can be a sequence with __iter__

    Methods:
    .append(*data, **kwargs): add data to the array, where `data` fits format
            above, OR keywords can be used
    .keys() ~= dict().keys()
    .items() ~= dict().items()
    .values() ~= dict().values()
    .dict() returns dict version of .items()

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
    DictArray_default_keys = _ascii_fields

    def __init__(self, data=None, keys=None):
        logger.debug(f"__init__ {data}, rename={keys}")
        self.default_keys = self.DictArray_default_keys.copy()
        self._keys = ()
        self._map_key_to_index = dict()
        self._rename_dict = OrderedDict()

        array, parse_keys = self.parse_data_with_keys(data)
        logger.debug(f"__init__ keys={parse_keys}, shape={array.shape}, ndim={array.ndim}, {repr(array)}")

        extra_keys = OrderedDict()
        final_keys = []
        keys = keys or OrderedDict()
        if parse_keys and keys:
            logger.debug(f"renaming to {keys}")
            if isinstance(keys, dict):
                logger.debug(f"__init__ rename from dict")
                rename_copy = OrderedDict(keys)
                for p in parse_keys or list(rename_copy.keys()):
                    logger.debug(f"__init__ renaming {p}->?")
                    final_keys.append(rename_copy.pop(p, p))
                extra_keys = rename_copy
                logger.debug(f"__init__ rename={keys}, final_keys={final_keys}, extra_keys={extra_keys}")
            else:
                rename_dict = OrderedDict()
                rename_copy = list(keys)
                for i, p in enumerate(parse_keys or keys):
                    rename_dict[p] = keys[i]
                    rename_copy.remove(keys[i])
                keys = rename_dict
                final_keys = list(keys.values())
                extra_keys = {k:k for k in rename_copy}
                logger.debug(f"__init__ rename={keys}, final_keys={final_keys}, extra_keys={extra_keys}")
        elif parse_keys and not keys:
            logger.debug(f"builtin names")
            final_keys = parse_keys
        elif not parse_keys and keys:
            logger.debug(f"specified names")
            if isinstance(keys, dict):
                extra_keys = keys.copy()
                list(map(extra_keys.pop, list(keys.keys())))
                final_keys = tuple(keys.values())
            else:
                final_keys = tuple(keys)
                keys = OrderedDict((k, k) for k in keys)
        else:  # no data keys and no _rename_dict
            logger.debug(f"no keys at all")
            pass

        # do extra_keys
        if extra_keys:
            array = np.column_stack([array, np.full([array.shape[0], len(extra_keys)], np.nan)])
            keys.update(extra_keys)
            final_keys = tuple([*final_keys, *extra_keys.values()])

        logger.debug(f"__init__ final keys {final_keys}")
        utils.remove_keys(self, final_keys)  # update ascii list
        utils.finish_keys(self, final_keys)  # store _keys and _map_key_to_index
        logger.debug(f"__init__ _keys {self._keys} -> {self._map_key_to_index}")
        self._rename_dict.update(keys)
        self.array = array

    def parse_data_with_keys(self, data):
        array, parse_keys = parse_data(data)
        data_has_keys = bool(parse_keys) or len(array.shape) > 1
        if data_has_keys and not parse_keys:  # make automatic names
            parse_keys = self.default_keys[:array.shape[1]]
        if self._rename_dict:  # generate new keys from _rename_dict
            parse_keys = tuple(map(lambda k: self._rename_dict.get(k, k), parse_keys))
        return array, parse_keys

    def rename(self, input_key, new_access_key):
        logger.info(f"rename {input_key}, {new_access_key}")
        if new_access_key in self._rename_dict.values():
            raise ValueError(f"Key `{new_access_key}` already assigned")
        old_access_key = self._rename_dict.get(input_key, None)
        logger.info(f"rename old_access_key = {old_access_key}")

        def replace(oldkey, newkey, keys):
            return list(k if k != oldkey else newkey for k in keys)

        # if input_key in self._keys:
        if old_access_key is None:
            old_access_key = input_key
            newkeys = replace(input_key, new_access_key, self._keys)
        else:
            newkeys = replace(old_access_key, new_access_key, self._keys)
        logger.info(f"rename old_access_key = {old_access_key}, newkeys= {newkeys}")

        if old_access_key in self.DictArray_default_keys:
            logger.info(f"rename add old_access_key back into list")
            self.default_keys.append(new_access_key)

        if new_access_key not in newkeys:
            newkeys = (*self._keys, new_access_key)
            logger.info(f"rename extend keys {newkeys}")
            # extend

        self._rename_dict[input_key] = new_access_key

        utils.remove_keys(self, new_access_key)
        utils.finish_keys(self, newkeys)

    # def extend(self, data_dict):
    #     array, parse_keys = self.parse_data_with_keys(data_dict)
    #     for k in parse_keys:
    #         if k in self._keys:
    #             raise ValueError(f"Key {k} already assigned")
    #         self._rename_dict[k] = k
    #     newkeys = (*self._keys, *self._rename_dict.keys())
    #     utils.remove_keys(self, newkeys)
    #     utils.finish_keys(self, newkeys)
    #     # self.array

    # def append(self, *data, **data_kw):
    #     """Process and add data points
    #
    #     :param data: data in format supported by DictArray()
    #     :param data_kw: keywords specifying data
    #     :return: np.ndarray: the processed, appended data
    #     """
    #     if not data and not data_kw:
    #         raise ValueError("No data provided")
    #     elif not data and data_kw:
    #         data = data_kw
    #     elif len(data) == 1:
    #         data = data[0]
    #
    #     data, keys = parse_data(data)
    #     s = self.array.shape
    #     if self.array.size == 0 and s[1] == 0:  # no data, no keys yet
    #         if data.size > 0 or data.shape[1] > 0:  # new data not useless
    #             self.array.resize(data.shape)
    #             self.array[:] = data[:]
    #             if keys:
    #                 utils.remove_keys(self, keys)
    #                 utils.finish_keys(self, keys)
    #             else:
    #                 utils.finish_keys(self, utils.generate_keys(self, data.shape[1]))
    #     else:  # have keys set
    #         if keys:  # new data has keys
    #             if set(keys) == set(self._keys):  # keys are same
    #                 if data.size == 0:  # no actual new data
    #                     return data
    #                 if keys == self._keys:  # keys are same, in same order
    #                     if data.size == 0:  # no actual new data
    #                         return data
    #                     sdata = data  # set as sorted
    #                 else:  # keys in different order
    #                     keys = list(keys)  # so we can use .index
    #                     sdata = np.ones_like(data)  # sorted data
    #                     for i, k in enumerate(self._keys):  # sort incoming data columns
    #                         sdata[:, i] = data[:][keys.index(k)]
    #
    #                 # actually append data
    #                 self.array.resize(s[0] + sdata.shape[0], s[1])
    #                 # self.array[s[0]:, :] = sdata[:]
    #                 np.concatenate([self.array, data], 0, out=self.array)
    #         else:
    #             np.concatenate([self.array, data], 0, out=self.array)
    #     return data

    # <editor-fold> convenience funcs
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

    # </editor-fold>

    # <editor-fold> getitem & setitem
    def _process_item_key(self, key, axis=0):
        # TODO: add float handling
        if not utils.contains_non_index(key):
            return key
        if isinstance(key, slice):
            return utils.do_slice(self, key)
        elif isinstance(key, tuple):
            return tuple(self._process_item_key(k, axis) for k in key)
        elif isinstance(key, list):
            return list(self._process_item_key(k, axis) for k in key)
        else:
            return self._map_key_to_index[key]

    def __getitem__(self, key):
        # TODO: add float handling
        logger.info(f"getitem `{key}`")
        if utils.contains_non_index(key):
            if isinstance(key, tuple):  # ['x'] OK // NOT OK ['x', 0]
                if utils.contains_non_index(key[0]) and len(key) > 1:
                    raise TypeError(f"custom keys cannot be used on axis 0 while indexing other axes")
                key = tuple(self._process_item_key(k, i) for i, k in enumerate(key))
            elif isinstance(key, slice):  # ['x':'z']  [0:'y']  ['x':2:1] OK // NOT OK [~:~:'x']
                key = slice(None), utils.do_slice(self, key)
            else:
                key = slice(None), self._process_item_key(key)
        return self.array[key]

    def __setitem__(self, key, value):
        # TODO: add float handling
        logger.info(f"getitem `{key}`")
        if utils.contains_non_index(key):
            if isinstance(key, tuple):  # ['x'] OK // NOT OK ['x', 0]
                if utils.contains_non_index(key[0]) and len(key) > 1:
                    raise TypeError(f"custom keys cannot be used on axis 0 while indexing other axes")
                key = tuple(self._process_item_key(k, i) for i, k in enumerate(key))
            elif isinstance(key, slice):  # ['x':'z']  [0:'y']  ['x':2:1] OK // NOT OK [~:~:'x']
                key = slice(None), utils.do_slice(self, key)
            else:
                key = slice(None), self._process_item_key(key)
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
    names = None  # keys to return
    logger.debug(f"parse_data `{repr(data)}`")
    if data is None:
        return np.array([]), names
    dt = data_type(data)
    logger.debug(f"dt={dt}")
    if dt == 'empty':
        logger.info(f"DOING empty {repr(data)}")
        return np.array([]), names
    elif dt == 'listOfLists':
        data_fields = data._fields if hasattr(data, '_fields') else None
        sample_fields = data[0]._fields if hasattr(data[0], '_fields') else None
        data = list(map(list, data))
        names = data_fields or sample_fields or None
        logger.debug(f"listOfLists {repr(data)}")
        if data_fields:  # lists are in named tuple
            data = np.array(data).T
            logger.debug(f"listOfLists {repr(data)}")
        else:
            data = np.array(data)
        logger.debug(f"listOfLists -> {repr(data)}")
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
        return np.array(vvv), tuple(names)
    elif dt == 'listOfValues':
        if hasattr(data, '_fields'):  # namedtuple
            names = data._fields
        return np.array([[*data]]), names
    elif dt == 'dictOfLists':
        names, data = tuple(data.keys()), list(map(lambda *a: list(a), *data.values()))  # transpose data
        data = np.array(data)
        logger.debug(f'dictOfLists {names}, {repr(data)}')
        if data.size == 0:
            logger.debug('make to size')
            data.resize((0, len(names)))
        return data, names
    elif dt == 'dictOfValues':
        names, data = tuple(data.keys()), [list(data.values())]
        logger.debug(f"dictOfValues {repr(data)}")
        data = np.array(data)
        return np.array(data), names
    elif dt == 'recarray':
        names = data.dtype.names
        logger.debug(f"recarray {repr(data)}")
        if data.size == 0:
            data = np.ndarray((0, len(names)))
        else:
            data = np.asanyarray(data.tolist())  # get list version to remove dtype // np.void
        logger.debug(f"recarray {repr(data)}")
        return data, names
    elif dt == 'ndarray':
        return data.copy(), names
    elif dt == 'DictArray':
        return data.array.copy(), data._keys
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
    raise TypeError(f"Unknown data_type, {repr(obj)}")


class DataFileIO():
    @staticmethod
    def parse_file(filename):
        logger.debug(f"parse_file `{filename}`")
        data, ff = DataFileIO.detect_format(filename)

        if data is not None:
            logger.debug((data, ff))
            return data, ff

        if ff['file'] == 'empty':
            logger.debug((data, ff))
            return data, ff
        elif ff['file'] == 'numpy':
            data = DataFileIO.do_numpy_csv(ff, filename)
        elif ff['file'] is None:
            try:
                data = json.load(open(filename))
            except json.JSONDecodeError as e:
                logger.debug(f"Doing numpy {ff}")
                data = DataFileIO.do_numpy_csv(ff, filename)
        elif ff['file'] == 'multiLine':
            logger.debug(f"Doing multiLine {ff}")
            data = DataFileIO.do_multiline_literal(ff, filename)
        else:
            raise NotImplementedError(ff)

        logger.debug((data, ff))
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
        logger.debug(f"detect_format {filename}")
        file = open(filename)
        ff = {'data': None, 'file': None, 'header': None, 'dialect': None}

        try:
            samplelines = DataFileIO.readnlines(file, 4)
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

        logger.debug('here')
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
                    file.seek(0)
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

        logger.debug(ff)

        if ff['file'] == 'oneLine':
            logger.debug((sample, ff))
            return sample, ff
        logger.debug((None, ff))
        return None, ff

    @staticmethod
    def do_multiline_literal(ff, filename):
        logger.debug(filename)
        file = open(filename)

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
    def do_numpy_csv(ff, filename):
        try:
            data = np.load(filename)
            ff['data'] = 'save'
            ff['file'] = 'numpy'
        except ValueError:
            kw = {'fname': filename}
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
