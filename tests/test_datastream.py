import pytest
import os
import logging
from collections import namedtuple
import numpy as np

from datastream import DataStream, data_type

logging.basicConfig(filename='logs/test_datastream.log', filemode='w', level=logging.INFO)

# test data
dict1 = {'x': [], 'y': []}
dict2 = {'x': 1.0, 'y': 1.0}
dict3 = {'x': [2.0, 3.0], 'y': [2.0, 3.0]}
point_test = namedtuple('point_test', 'x, y')
ntup1 = point_test([], [])
ntup2 = point_test(1.0, 1.0)
ntup3 = point_test([2.0, 3.0], [2.0, 3.0])
tup1 = ([], [])
tup2 = (1.0, 1.0)
tup3 = ([2.0, 3.0], [2.0, 3.0])
np1 = (np.array([]), np.array([]))
np2 = (np.array(1.0), np.array(1.0))
np3 = (np.array([2.0, 3.0]), np.array([2.0, 3.0]))
VALUES1 = np.array([[],[]])
VALUES2 = np.array([[1.0],[1.0]])
VALUES3 = np.array([[2.0,3.0],[2.0,3.0]])
KEYS = {'x', 'y'}


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


def test_dataType():
    logging.debug('---------------Begin test_dataType()')
    assert data_type(DataStream()) == 'DictArray'
    assert data_type(DataStream([[], [], []])) == 'DictArray'


def test_record_file_csv():
    logging.debug('---------------Begin test_record_file_csv()')

    file = './logs/datastream_test_record_file_csv.log'
    try: os.remove(file)
    except: pass

    dict2 = {'x': 1.0, 'y': 1.0}

    ds = DataStream(record_to_file=file)

    ds.append(dict2)
    assert open(file, 'r').read() == (os.linesep).join(['x,y', '1.0,1.0'])+os.linesep

    ds.append(dict2)
    assert open(file, 'r').read() == (os.linesep).join(['x,y', '1.0,1.0', '1.0,1.0'])+os.linesep


def test_record_file_dict():
    logging.debug('---------------Begin test_record_file_dict()')

    file = './logs/datastream_test_record_file_dict.log'
    try: os.remove(file)
    except: pass

    dict2 = {'x': 1.0, 'y': 1.0}

    ds = DataStream(record_to_file=file, file_format='dict')
    ds.append(dict2)
    assert str({'x':1.0,'y':1.0}) + os.linesep == open(file, 'r').readlines()[-1]

    ds.append(dict2)
    assert str({'x':1.0,'y':1.0}) + os.linesep == open(file, 'r').readlines()[-1]


def test_record_file_list():
    logging.debug('---------------Begin test_record_file_list()')

    file = './logs/datastream_test_record_file_list.log'
    try: os.remove(file)
    except: pass

    dict2 = {'x': 1.0, 'y': 1.0}

    ds = DataStream(record_to_file=file, file_format='list')
    ds.append(dict2)
    assert 'x,y' + os.linesep + str(list(dict2.values())) + os.linesep == open(file, 'r').read()
    assert str(list(dict2.values())) + os.linesep == open(file, 'r').readlines()[-1]

    ds.append(dict2)
    assert str(list(dict2.values())) + os.linesep == open(file, 'r').readlines()[-1]
