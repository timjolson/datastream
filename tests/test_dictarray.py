import pytest
import os
import logging
from collections import namedtuple
import numpy as np

from datastream import DictArray, data_type

logging.basicConfig(filename='logs/test_DictArray.log', filemode='w', level=logging.INFO)

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


def check_equal(one, two):
    logging.debug('check_equal')
    logging.debug(f"keys {one._keys, two._keys}")
    logging.debug(f"points {one.points, two.points}")

    assert set(one._keys) == set(two._keys)
    assert np.all(one.points == two.points)


def test_base_constructor():
    logging.debug('---------------Begin test_base_constructor()')
    da = DictArray()

    # kwargs passed to np.array
    da = DictArray(dtype=np.float64)
    assert da.dtype == np.float64


def test_dataType():
    logging.debug('---------------Begin test_dataType()')

    assert data_type(dict1) == 'dictOfLists'
    assert data_type(dict2) == 'dictOfValues'

    assert data_type(ntup1) == 'listOfLists'
    assert data_type(ntup2) == 'listOfValues'

    assert data_type(tup1) == 'listOfLists'
    assert data_type(tup2) == 'listOfValues'

    assert data_type(np1) == 'listOfLists'
    assert data_type(np2) == 'listOfLists'

    assert data_type(DictArray()) == 'empty'
    assert data_type(DictArray([])) == 'empty'
    assert data_type(DictArray([[]])) == 'DictArray'
    assert data_type(DictArray([[], [], []])) == 'DictArray'


def test_constructor_point():
    logging.debug('---------------Begin test_constructor_point()')

    for data in [dict1, ntup1, tup1, np1]:
        da = DictArray(data)
        assert data_type(da) == 'DictArray'
        assert np.all(da.points == VALUES1)
        assert set(da.keys()) == KEYS

    for data in [dict2, ntup2, tup2, np2]:
        da = DictArray(data)
        assert data_type(da) == 'DictArray'
        assert np.all(da.points == VALUES2)
        assert set(da.keys()) == KEYS


def test_constructor_points():
    logging.debug('---------------Begin test_constructor_points()')

    for data in [dict3, ntup3, tup3, np3]:
        da = DictArray(data)
        assert data_type(da) == 'DictArray'
        assert np.all(da.points == VALUES3)
        assert set(da.keys()) == KEYS


def test_array_read_ops():
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    da = DictArray(Dict)

    # check by point
    assert np.all(da[:,0] == [1., 2., 3])
    assert np.all(da[:,1] == [4., 5., 6])

    # check x
    assert np.all(da['x'] == da['x',:])
    assert np.all(da['x'] == da[0,:])
    assert np.all(da['x'] == [1,4])
    assert da['x',0] == 1 == da['x'][0]
    assert da['x',1] == 4 == da['x'][1]

    # check y
    assert np.all(da['y'] == da['y',:])
    assert np.all(da['y'] == da[1,:])
    assert np.all(da['y'] == [2, 5])
    assert da['y',0] == 2 == da['y'][0]
    assert da['y',1] == 5 == da['y'][1]

    #check z
    assert np.all(da['z'] == da['z',:])
    assert np.all(da['z'] == da[2,:])
    assert np.all(da['z'] == [3, 6])
    assert da['z',0] == 3 == da['z'][0]
    assert da['z',1] == 6 == da['z'][1]

    with pytest.raises(IndexError):
        print(da[0,'z'])
    with pytest.raises(IndexError):
        print(da[:,'x'])

    check_equal(eval(repr(da), {}, {'DictArray':DictArray, 'array':np.array}), da)


def test_array_set_ops():
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    ds_constructed = DictArray(Dict)

    # [key] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da['x'] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da['y'] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed.points == da.points) == False  # 'is' doesn't want to work here

    da['z'] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed.points == da.points)

    # [key, :] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da['x',:] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da['y',:] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed.points == da.points) == False  # 'is' doesn't want to work here

    da['z',:] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed.points == da.points)

    # [key][:] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da['x'][:] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da['y'][:] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed.points == da.points) == False  # 'is' doesn't want to work here

    da['z'][:] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed.points == da.points)

