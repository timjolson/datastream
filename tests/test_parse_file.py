import pytest
import os, sys
import numpy as np
import logging
from datastream import parse_file, parse_data
from . import loadfileformats, filenames

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(name)s:%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')

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
