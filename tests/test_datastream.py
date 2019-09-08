import pytest
import os
import logging
import numpy as np
import sys

from datastream import DataStream, data_type
from . import filenames

# logging.basicConfig(filename='logs/test_datastream.log', filemode='w', level=logging.DEBUG)
TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
logging.basicConfig(stream=sys.stdout, format=TRACEBACK_FMT+'%(message)s', filemode='w', level=logging.DEBUG)


def add_point_tester(ds):
    # using np.inf instead of None or np.nan so we can check equality between arrays

    logging.debug('addpoints dict')
    ds.append({'x': 0, 'y': 0, 'z': 0})
    logging.debug(ds)
    logging.debug('addpoints items')
    ds.append(1, np.inf, 1)
    logging.debug(ds)
    logging.debug('addpoints kwargs')
    ds.append(x=2, y=1, z=np.inf)
    logging.debug(ds)
    logging.debug('addpoints tuple')
    ds.append((3, 2, 2))
    logging.debug(ds)


def check_post_add_point(ds):
    # using np.inf instead of None or np.nan so we can check equality between arrays

    logging.debug('post addpoints')
    assert np.array_equal(ds['x'], [0, 1, 2, 3])
    logging.debug('post addpoints')
    assert np.array_equal(ds['y'], np.array([0., np.inf, 1., 2.]))
    logging.debug('post addpoints')
    assert np.array_equal(ds['z'], np.array([0., 1., np.inf, 2.]))


def check_datastreams_equal(one, two):
    logging.debug('check_datastreams_equal')
    logging.debug(f"keys {one._keys, two._keys}")
    logging.debug(f"points {one.array, two.array}")

    assert set(one._keys) == set(two._keys)
    assert np.all(one.array == two.array)


def test_base_constructor():
    logging.debug('---------------Begin test_base_constructor()')
    ds = DataStream()


def test_construct_equal():
    ds1 = DataStream()
    add_point_tester(ds1)
    ds2 = DataStream(ds1)
    check_datastreams_equal(ds1, ds2)


def test_data_type():
    logging.debug('---------------Begin test_dataType()')
    assert data_type(DataStream()) == 'DictArray'
    assert data_type(DataStream([[], [], []])) == 'DictArray'


def test_load_file():
    logging.debug('---------------Begin test_load_file()')

    # data_set_dir = 'data_sets'
    # filenames = [os.path.join(data_set_dir, f) for f in os.listdir(data_set_dir) if not f.startswith('.')]

    for file in filenames:
        logging.debug(f'Loading {file}')
        ds = DataStream(file)
        logging.debug(f'Data is {ds}')

        if not file.endswith('empty'):
            assert np.array_equal(ds, [[4, 8, 0], [5, 9, 1], [6, 10, 2], [7, 11, 3]])

        if file.find('header') != -1 or file.find('structured') != -1 or file.find('dict') != -1:
            assert ds._keys == ('x', 'y', 'time')
        elif file.endswith('empty'):
            assert ds.array.size == 0
            assert ds._keys == ()
        else:
            assert ds._keys == ('x', 'y', 'z')


def test_load_append_text_source():
    logging.debug('---------------Begin test_load_append_text_source()')

