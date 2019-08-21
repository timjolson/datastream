import pytest
import os, sys
import numpy as np
import logging
from datastream import parse_file, parse_data

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(name)s:%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')

time, x, y = [0,1,2,3], [4,5,6,7], [8,9,10,11]
data_set_dir = 'data_sets'
header = ('x', 'y', 'time')
loadfileformats = {
    'csv':                      {'file':'csv', 'data':'text', 'header':None},
    'csv_with_header':          {'file':'csv', 'data':'text', 'header':('x', 'y', 'time')},
    'dict_of_lists_multiline':  {'file':'multiLine', 'data':'dictOfLists', 'header':None},
    'dict_of_lists_oneline':    {'file':'oneLine', 'data':'dictOfLists', 'header':None},
    'empty':                    {'file':'empty', 'data':None, 'header':None},
    'json_dump_dict_of_lists':  {'file':'oneLine', 'data':'dictOfLists', 'header':None},
    'json_dump_list_of_dicts':  {'file':'oneLine', 'data':'listOfDicts', 'header':None},
    'json_dump_list_of_lists':  {'file':'oneLine', 'data':'listOfLists', 'header':None},
    'list_of_dicts_oneline':    {'file':'oneLine', 'data':'listOfDicts', 'header':None},
    'list_of_lists_multiline':  {'file':'multiLine', 'data':'listOfLists', 'header':None},
    'list_of_lists_oneline':    {'file':'oneLine', 'data':'listOfLists', 'header':None},
    'np_save_arr.npy':          {'file':'numpy', 'data':'save', 'header':None},
    'np_save_structuredarray.npy':  {'file':'numpy', 'data':'save-structuredarray', 'header':('x', 'y', 'time')},
    'np_savetxt_arr':           {'file':'csv', 'data':'text', 'header':None},
    'np_savetxt_arr_header':    {'file':'csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_savetxt_arr_header_delim':  {'file':'csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_savetxt_structarr':     {'file':'csv', 'data':'text', 'header':None},
    'np_savetxt_structarr_header':  {'file':'csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_savetxt_structarr_header_delim':    {'file':'csv', 'data':'text', 'header':('x', 'y', 'time')},
    'some_dicts_multiline':     {'file':'multiLine', 'data':'listOfDicts', 'header':None},
    'some_dicts_oneline':       {'file':'oneLine', 'data':'listOfDicts', 'header':None},
    # 'some_lists_header':        {'file':'multiLine', 'data':'listOfValues', 'header':('x', 'y', 'time')},
    'some_lists_multiline':     {'file':'multiLine', 'data':'listOfValues', 'header':None},
    'some_lists_oneline':       {'file':'oneLine', 'data':'listOfLists', 'header':None},
}
filenames = [os.path.join(data_set_dir,f) for f in loadfileformats]
for f in list(loadfileformats.keys()):
    path = os.path.join(data_set_dir,f)
    loadfileformats[path] = loadfileformats[f]
    del loadfileformats[f]

VALUES = np.array([[4,8,0],[5,9,1],[6,10,2],[7,11,3]])
KEYS = ('x', 'y', 'time')


def test_parse_file():
    for f in filenames:
        logging.info(f"filename={f}")
        d, ff = parse_file(f)
        ff.pop('dialect', None)
        assert ff == loadfileformats[f]

        r, keys, nkeys = parse_data(d)
        if f.endswith('empty'):  # empty test
            assert r.size == 0
        else:
            logging.debug(f"r.shape={r.shape}")
            assert np.array_equal(r, VALUES)  # actual data is same
            if f.find('dict') != -1 or f.find('header') != -1 or f.find('save_struct') != -1:
                assert keys == KEYS  # should have correct keys based on file name
            else:
                assert keys is ()  # uses default keys order
