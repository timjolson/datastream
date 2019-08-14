import pytest
import os, sys
import logging
import numpy as np
from datastream import DictArray, data_type

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')
# logging.basicConfig(filename='tests/logs/test_DictArray.log', filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')


from . import empty, dictOfLists, dictOfValues, listOfDicts, \
    listOfLists, listOfValues, ndarray, recarray, VALUES, \
    NAMEDTUPLEKEYS, RECARRAYKEYS, DICTKEYS, KEYS, groups


def check_equal(one, two):
    logging.debug('check_equal')
    logging.debug(f"keys {one._keys, two._keys}")

    assert set(one._keys) == set(two._keys)
    assert np.array_equal(one.array, two.array)
    assert np.array_equal(one, two)


def test_base_constructor():
    logging.info('---------------Begin test_base_constructor()')
    da = DictArray()

    check_equal(eval(repr(da), {}, {'DictArray':DictArray, 'array':np.array}), da)


def test_data_type():
    logging.info('---------------Begin test_data_type()')
    assert data_type(DictArray()) == 'DictArray'
    assert data_type(DictArray([])) == 'DictArray'
    assert data_type(DictArray([[]])) == 'DictArray'
    assert data_type(DictArray([[], []])) == 'DictArray'


def test_constructor_data():
    logging.info('---------------Begin test_constructor_data()')

    for groupname, group in groups.items():
        for dataname, data in group.items():
            logging.info(f"Testing `{groupname}` : `{dataname}` : {repr(data)}")
            ds = DictArray(data)

            vcheck = VALUES[dataname[-1]]
            if groupname.lower().find('dict') != -1:
                kcheck = DICTKEYS
            elif groupname.lower().find('rec') != -1:
                kcheck = RECARRAYKEYS
            elif dataname.lower().find('ntup') != -1:
                kcheck = NAMEDTUPLEKEYS
            elif dataname.lower().find('empty') != -1:
                kcheck = ()
            else:
                kcheck = KEYS

            # override for when using [] // [[]] // [[],[]]
            if ds.ndim <= 1:
                kcheck = ()
            else:
                if ds.shape[1] == 0:
                    kcheck = ()

            logging.info(f"kcheck={kcheck}, vcheck={vcheck}")
            logging.info(f"result={ds._keys, ds.array}")
            assert np.array_equal(ds.array, vcheck)
            assert ds._keys == kcheck


def test_array_read_ops():
    logging.info('---------------Begin test_array_read_ops()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    da = DictArray(Dict)

    # check by point
    assert np.all(da[0] == [1., 2., 3])
    assert np.all(da[1] == [4., 5., 6])

    # check x
    assert np.all(da['x'] == da[:, 'x'])
    assert np.all(da['x'] == da[:, 0])
    assert np.all(da['x'] == [1,4])
    assert da[0, 'x'] == 1 == da['x'][0]
    assert da[1, 'x'] == 4 == da['x'][1]

    # check y
    assert np.all(da['y'] == da[:, 'y'])
    assert np.all(da['y'] == da[:, 'y'])
    assert np.all(da['y'] == [2, 5])
    assert da[0,'y'] == 2 == da['y'][0]
    assert da[1,'y'] == 5 == da['y'][1]

    #check z
    assert np.all(da['z'] == da[:,'z'])
    assert np.all(da['z'] == da[:,2])
    assert np.all(da['z'] == [3, 6])
    assert da[0,'z'] == 3 == da['z'][0]
    assert da[1,'z'] == 6 == da['z'][1]

    with pytest.raises(TypeError):
        print(da['z',0])
    with pytest.raises(KeyError):
        print(da['a'])


def test_array_set_ops():
    logging.info('---------------Begin test_array_set_ops()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    ds_constructed = DictArray(Dict)

    # [key] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da['x'] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da['y'] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da['z'] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed == da)

    # [key, :] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da[:,'x'] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da[:,'y'] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da[:,'z'] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed == da)

    # [key][:] =
    da = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    da['x'][:] = Dict['x']
    assert np.all(ds_constructed['x'] == da['x'])

    da['y'][:] = Dict['y']
    assert np.all(ds_constructed['y'] == da['y'])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da['z'][:] = Dict['z']
    assert np.all(ds_constructed['z'] == da['z'])
    assert np.all(ds_constructed == da)


def test_getitem_multiple():
    logging.info('---------------Begin test_getitem_multiple()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}

    da = DictArray(Dict)
    logging.debug(da[['x','y']])
    assert np.all(da[['x', 'y']] == np.column_stack([da['x'], da['y']]))

    with pytest.raises(TypeError):
        da[('x', 'y'), 0]

    with pytest.raises(TypeError):
        da[['x', 'y'], 0]


def test_setitem_multiple():
    logging.info('---------------Begin test_setitem_multiple()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}

    da = DictArray(Dict)
    replace = np.column_stack([da['y'], da['x']]).copy()
    da[['x', 'y']] = replace

    assert np.all(da[['x', 'y']] == [[2.,1.],[5.,4.]])
    assert np.all(da[['x', 'y']] == replace)


def test_object_keys():
    logging.info('---------------Begin test_object_keys()')
    class thing(object):
        pass
    x, y, z = thing(), thing(), thing()
    Dict = {x: [1., 4.], y: [2., 5], z: [3, 6]}
    da = DictArray(Dict)

    # check by point
    assert np.all(da[0,:] == [1., 2., 3])
    assert np.all(da[1,:] == [4., 5., 6])

    # check x
    assert np.all(da[x] == da[:, x])
    assert np.all(da[x] == da[:, 0])
    assert np.all(da[x] == [1, 4])
    assert da[0, x] == 1 == da[x][0]
    assert da[1, x] == 4 == da[x][1]

    # check y
    assert np.all(da[y] == da[:, y])
    assert np.all(da[y] == da[:, 1])
    assert np.all(da[y] == [2, 5])
    assert da[0, y] == 2 == da[y][0]
    assert da[1, y] == 5 == da[y][1]

    # check z
    assert np.all(da[z] == da[:, z])
    assert np.all(da[z] == da[:, 2])
    assert np.all(da[z] == [3, 6])
    assert da[0, z] == 3 == da[z][0]
    assert da[1, z] == 6 == da[z][1]

    with pytest.raises(TypeError):
        print(da[z, 0])
    with pytest.raises(TypeError):
        print(da[x, :])

    ds_constructed = DictArray(Dict)

    # [key] =
    da = DictArray({x: [0, 0], y: [0, 0], z: [0, 0]})
    da[x] = Dict[x]
    assert np.all(ds_constructed[x] == da[x])

    da[y] = Dict[y]
    assert np.all(ds_constructed[y] == da[y])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da[z] = Dict[z]
    assert np.all(ds_constructed[z] == da[z])
    assert np.all(ds_constructed == da)

    da = DictArray({x: [0, 0], y: [0, 0], z: [0, 0]})
    da[:, x] = Dict[x]
    assert np.all(ds_constructed[x] == da[x])

    da[:, y] = Dict[y]
    assert np.all(ds_constructed[y] == da[y])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da[:, z] = Dict[z]
    assert np.all(ds_constructed[z] == da[z])
    assert np.all(ds_constructed == da)

    da = DictArray({x: [0, 0], y: [0, 0], z: [0, 0]})
    da[x][:] = Dict[x]
    assert np.all(ds_constructed[x] == da[x])

    da[y][:] = Dict[y]
    assert np.all(ds_constructed[y] == da[y])

    assert np.all(ds_constructed == da) == False  # 'is' doesn't want to work here

    da[z][:] = Dict[z]
    assert np.all(ds_constructed[z] == da[z])
    assert np.all(ds_constructed == da)

