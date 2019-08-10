import pytest
import os
import numpy as np
from datastream import DataStream, DataFileIO

time, x, y = [0,1,2,3], [4,5,6,7], [8,9,10,11]
data_set_dir = 'data_sets'
header = ('x', 'y', 'time')
loadfileformats = {
    'csv':{'file':'numpy-csv', 'data':'text', 'header':None},
    'csv_with_header':{'file':'numpy-csv', 'data':'text', 'header':('x', 'y', 'time')},
    'dict_of_lists':{'file':'oneLine', 'data':'dictOfLists', 'header':None},
    'empty':{'file':'empty', 'data':None, 'header':None},
    'json_dump_dict_of_lists':{'file':'oneLine', 'data':'dictOfLists', 'header':None},
    'json_dump_list_of_dicts':{'file':'oneLine', 'data':'listOfDicts', 'header':None},
    'json_dump_list_of_lists':{'file':'oneLine', 'data':'listOfLists', 'header':None},
    'list_of_dicts':{'file':'oneLine', 'data':'listOfDicts', 'header':None},
    'list_of_lists':{'file':'oneLine', 'data':'listOfLists', 'header':None},
    'some_dicts':{'file':'multiLine', 'data':'listOfDicts', 'header':None},
    'somedicts_oneline':{'file':'oneLine', 'data':'listOfDicts', 'header':None},
    'some_lists':{'file':'multiLine', 'data':'listOfValues', 'header':None},
    'somelists_oneline':{'file':'oneLine', 'data':'listOfLists', 'header':None},
    'listoflists_multiline':{'file':'multiLine', 'data':'listOfLists', 'header':None},
    'np_save_arr.npy':{'file':'numpy', 'data':'save', 'header':None},
    'np_savetxt_arr':{'file':'numpy-csv', 'data':'text', 'header':None},
    'np_savetxt_arr_header':{'file':'numpy-csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_savetxt_arr_header_delim':{'file':'numpy-csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_save_structuredarray.npy':{'file':'numpy', 'data':'save-structuredarray', 'header':('x', 'y', 'time')},
    'np_savetxt_structuredarray':{'file':'numpy-csv', 'data':'text', 'header':None},
    'np_savetxt_structuredarray_header':{'file':'numpy-csv', 'data':'text', 'header':('x', 'y', 'time')},
    'np_savetxt_structuredarray_header_delim':{'file':'numpy-csv', 'data':'text', 'header':('x', 'y', 'time')},
}
filenames = [os.path.join(data_set_dir,f) for f in loadfileformats]
for f in list(loadfileformats.keys()):
    path = os.path.join(data_set_dir,f)
    loadfileformats[path] = loadfileformats[f]
    del loadfileformats[f]

ds = DataStream({'x':x,'y':y,'time':time})
ds_keys = set(ds.keys())


def test_parse_format():
    for f, ff in loadfileformats.items():
        ff.pop('dialect', None)
        res = DataFileIO.parse_file(f)[1]
        res.pop('dialect', None)
        assert res == ff


def test_parse_file():
    for f in filenames:
        r = DataStream(f)
        if not f.endswith('empty'):
            assert np.array_equal(r, ds)
            if f.find('dict') != -1 or f.find('header') != -1 or f.find('save_struct') != -1:
                assert set(r.keys()) == ds_keys
            else:
                assert set(r.keys()) == set(['x', 'y', 'z'])
        else:
            assert r.size == 0
