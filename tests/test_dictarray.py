import pytest
import os, sys
import logging
import numpy as np
from collections import OrderedDict
from datastream import DictArray, data_type

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')
# logging.basicConfig(filename='tests/logs/test_DictArray.log', filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')


# data tests
from . import empty, dictOfLists, dictOfValues, listOfDicts, \
    listOfLists, listOfValues, ndarray, recarray, VALUES, \
    NAMEDTUPLEKEYS, RECARRAYKEYS, DICTKEYS, KEYS, groups


def add_point_tester(da):
    # using np.inf instead of None or np.nan so we can check equality between arrays

    logging.debug('addpoints dict')
    da.append({'x': 0, 'y': 0, 'z': 0})
    logging.debug(da)
    logging.debug('addpoints items')
    da.append(1, np.inf, 1)
    logging.debug(da)
    logging.debug('addpoints kwargs')
    da.append(x=2, y=1, z=np.inf)
    logging.debug(da)
    logging.debug('addpoints tuple')
    da.append((3, 2, 2))
    logging.debug(da)


def check_post_add_point(da):
    # using np.inf instead of None or np.nan so we can check equality between arrays

    logging.debug('post addpoints')
    assert np.array_equal(da['x'], [0, 1, 2, 3])
    logging.debug('post addpoints')
    assert np.array_equal(da['y'], np.array([0., np.inf, 1., 2.]))
    logging.debug('post addpoints')
    assert np.array_equal(da['z'], np.array([0., 1., np.inf, 2.]))
    assert da.array.shape == (4,3)
    assert da._keys == ('x', 'y', 'z')


def check_equal(one, two):
    logging.debug('check_equal')
    logging.debug(f"one {repr(one)}")
    logging.debug(f"two {repr(two)}")
    logging.debug(f"keys {one._keys, two._keys}")
    logging.debug(f"data {one.array, two.array}")

    assert set(one._keys) == set(two._keys)
    assert np.array_equal(one.array, two.array)
    assert np.array_equal(one, two)


def test_base_constructor():
    logging.info('---------------Begin test_base_constructor()')
    da = DictArray()
    assert data_type(DictArray()) == 'DictArray'


def test_constructor_data():
    logging.info('---------------Begin test_constructor_data()')

    for groupname, group in groups.items():
        for dataname, data in group.items():
            logging.info(f"Testing `{groupname}` : `{dataname}` : {repr(data)}")
            da = DictArray(data)

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
            if da.ndim == 0:
                logging.info(f'OVERRIDE kcheck to () when no data')
                kcheck = ()
            elif da.ndim == 1:
                logging.info(f'OVERRIDE kcheck to () when no data')
                kcheck = ()
                vcheck = np.array([])
            else:
                if da.shape[1] == 0:
                    kcheck = ()
                    logging.info('OVERRIDE to kcheck to ()')

            logging.info(f"kcheck={kcheck}, vcheck={repr(vcheck)}")
            logging.info(f"result={da._keys, da.array}")
            assert np.array_equal(da.array, vcheck)
            assert da._keys == kcheck

    for groupname, group in groups.items():
        for dataname, data in group.items():
            da = DictArray(data)
            if da.size or da._keys:  # skip empty-ish tests
                logging.info(f"Repring `{groupname}` : `{dataname}` : {repr(data)}")
                check_equal(
                    eval(repr(da), {}, {'DictArray':DictArray, 'array':np.array}),
                    da)


def test_constructor_keys():
    logging.info('---------------Begin test_constructor_keys()')
    Dict = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]}
    List = [[1,2,3], [4,5,6]]

    auto_keys = tuple(Dict.keys())
    manual_keys = ('a', 'b', 'c')
    manual_keys_extra = ('a', 'b', 'c', 'D')
    rn_dict = dict(iter(zip(auto_keys, manual_keys)))
    rn_dict_extra_keys = rn_dict.copy()
    rn_dict_extra_keys.update(d='D')

    # providing keys will rename the incoming data
    assert DictArray(Dict, keys=rn_dict)._keys == manual_keys  # no extension
    assert DictArray(List, keys=rn_dict)._keys == manual_keys  # no extension
    # providing more keys than the data has will extend the DictArray to include the extras
    assert DictArray(Dict, keys=rn_dict_extra_keys)._keys == manual_keys_extra  # extension
    assert DictArray(List, keys=rn_dict_extra_keys)._keys == manual_keys_extra  # extension

    # provided no data, array will be sized to match len(keys)
    assert DictArray(keys=rn_dict)._keys == manual_keys
    assert DictArray(keys=rn_dict_extra_keys)._keys == manual_keys_extra
    assert DictArray({}, keys=manual_keys)._keys == manual_keys
    assert DictArray([], keys=manual_keys)._keys == manual_keys
    assert DictArray({}, keys=manual_keys_extra)._keys == manual_keys_extra
    assert DictArray([], keys=manual_keys_extra)._keys == manual_keys_extra
    assert DictArray([[],[]], keys=manual_keys)._keys == manual_keys
    assert DictArray([[],[]], keys=manual_keys_extra)._keys == manual_keys_extra

    da = DictArray(Dict, keys=manual_keys)
    assert da._keys == manual_keys
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])

    da = DictArray(List, keys=manual_keys)
    assert da._keys == manual_keys
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])

    da = DictArray(Dict, keys=manual_keys_extra)
    assert da._keys == manual_keys_extra
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])
    assert np.isnan(da['D'].sum())

    da = DictArray(List, keys=manual_keys_extra)
    assert da._keys == manual_keys_extra
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])
    assert np.isnan(da['D'].sum())

    da = DictArray(List, keys=rn_dict)
    assert da._keys == manual_keys
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])

    da = DictArray(List, keys=rn_dict_extra_keys)
    assert da._keys == manual_keys_extra
    assert np.array_equal(da['a'], Dict['x'])
    assert np.array_equal(da['b'], Dict['y'])
    assert np.array_equal(da['c'], Dict['z'])
    assert np.isnan(da['D'].sum())

    with pytest.raises(ValueError):
        da = DictArray(Dict, keys=('a', 'b'))


def test_array_read_ops():
    logging.info('---------------Begin test_array_read_ops()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    da = DictArray(Dict)

    # check by point
    assert np.all(da[0] == [1., 2., 3])
    assert np.all(da[1] == [4., 5., 6])

    # check index by key
    assert np.all(da['x'] == da[:, 'x'])
    assert np.all(da['x'] == da[:, 0])
    assert np.all(da['x'] == [1,4])
    assert da[0, 'x'] == 1 == da['x'][0]
    assert da[1, 'x'] == 4 == da['x'][1]

    # check slicing
    assert np.all(da['x':'y'] == [[1.,2.],[4.,5.]])
    assert np.all(da['x':'z':2] == [[1.,3.],[4.,6.]])
    assert np.all(da['y':'z'] == [[2.,3.],[5.,6.]])

    # check reverse slicing
    assert np.all(da['z':'y':-1] == [[3.,2.],[6.,5.]])
    assert np.all(da[:, 'z':'y':-1] == [[3.,2.],[6.,5.]])

    with pytest.raises(TypeError):
        print(da['z',0])
    with pytest.raises(KeyError):
        print(da['a'])


def test_array_set_ops():
    logging.info('---------------Begin test_array_set_ops()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}
    values = DictArray(Dict)

    # [key] =
    zeros = DictArray({'x':[0,0],'y':[0,0],'z':[0,0]})
    zeros['x'] = Dict['x']
    assert np.all(values['x'] == zeros['x'])
    zeros['y'] = Dict['y']
    assert np.all(values['y'] == zeros['y'])
    zeros['z'] = Dict['z']
    assert np.all(values['z'] == zeros['z'])
    # after setting all keys, should be identical
    assert np.all(values == zeros)

    # [:, key] =
    zeros = DictArray({'x': [0, 0], 'y': [0, 0], 'z': [0, 0]})
    zeros[:, 'x'] = Dict['x']
    assert np.all(values['x'] == zeros['x'])
    zeros[:, 'y'] = Dict['y']
    assert np.all(values['y'] == zeros['y'])
    zeros[:, 'z'] = Dict['z']
    assert np.all(values['z'] == zeros['z'])
    # after setting all keys, should be identical
    assert np.all(values == zeros)

    # [key][:] =
    zeros = DictArray({'x': [0, 0], 'y': [0, 0], 'z': [0, 0]})
    zeros['x'][:] = Dict['x']
    assert np.all(values['x'] == zeros['x'])
    zeros['y'][:] = Dict['y']
    assert np.all(values['y'] == zeros['y'])
    zeros['z'][:] = Dict['z']
    assert np.all(values['z'] == zeros['z'])
    # after setting all keys, should be identical
    assert np.all(values == zeros)


def test_getitem_multiple():
    logging.info('---------------Begin test_getitem_multiple()')
    Dict = {'x': [1., 4.], 'y': [2., 5], 'z': [3, 6]}

    da = DictArray(Dict)
    logging.debug(da[['x', 'y']])
    assert np.all(da[['x', 'y']] == np.column_stack([da['x'], da['y']]))
    assert np.all(da[['x', 'y']] == np.column_stack([Dict['x'], Dict['y']]))
    assert np.all(da[['x', 'y']] == [[1., 2.], [4, 5.]])

    # swap column order by indexing
    assert np.all(da[['y', 'x']] == [[2., 1.], [5., 4.]])
    assert np.all(da[['y', 'x']] == np.column_stack([Dict['y'], Dict['x']]))

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
    assert np.all(da[0] == [1., 2., 3])
    assert np.all(da[1] == [4., 5., 6])

    # check index by key
    assert np.all(da[x] == da[:, x])
    assert np.all(da[x] == da[:, 0])
    assert np.all(da[x] == [1, 4])
    assert da[0, x] == 1 == da[x][0]
    assert da[1, x] == 4 == da[x][1]

    # check slicing
    assert np.all(da[x:y] == [[1., 2.], [4., 5.]])
    assert np.all(da[x:z:2] == [[1., 3.], [4., 6.]])
    assert np.all(da[y:z] == [[2., 3.], [5., 6.]])

    # check reverse slicing
    assert np.all(da[z:y:-1] == [[3., 2.], [6., 5.]])
    assert np.all(da[:, z:y:-1] == [[3., 2.], [6., 5.]])

    with pytest.raises(TypeError):
        print(da[z, 0])
    with pytest.raises(KeyError):
        print(da['a'])

    # checking set ops
    Dict = {x: [1., 4.], y: [2., 5], z: [3, 6]}
    values = DictArray(Dict)

    # [key] =
    zeros = DictArray({x:[0,0],y:[0,0],z:[0,0]})
    zeros[x] = Dict[x]
    assert np.all(values[x] == zeros[x])
    zeros[y] = Dict[y]
    assert np.all(values[y] == zeros[y])
    zeros[z] = Dict[z]
    assert np.all(values[z] == zeros[z])
    # after setting all keys, should be identical
    assert np.all(values == zeros)

    # [:, key] =
    zeros = DictArray({x: [0, 0], y: [0, 0], z: [0, 0]})
    zeros[:, x] = Dict[x]
    assert np.all(values[x] == zeros[x])
    zeros[:, y] = Dict[y]
    assert np.all(values[y] == zeros[y])
    zeros[:, z] = Dict[z]
    assert np.all(values[z] == zeros[z])
    # after setting all keys, should be identical
    assert np.all(values == zeros)

    # [key][:] =
    zeros = DictArray({x: [0, 0], y: [0, 0], z: [0, 0]})
    zeros[x][:] = Dict[x]
    assert np.all(values[x] == zeros[x])
    zeros[y][:] = Dict[y]
    assert np.all(values[y] == zeros[y])
    zeros[z][:] = Dict[z]
    assert np.all(values[z] == zeros[z])
    # after setting all keys, should be identical
    assert np.all(values == zeros)


def test_rename():
    logging.info('---------------Begin test_rename()')
    Dict = {'x': [1, 4], 'y': [2, 5], 'z': [3, 6]}

    auto_keys = tuple(Dict.keys())
    manual_keys = ('a', 'b', 'c')
    manual_keys_extra = ('a', 'b', 'c', 'D')
    rn_dict = dict(iter(zip(auto_keys, manual_keys)))
    rn_dict_extra_keys = rn_dict.copy()
    rn_dict_extra_keys.update(d='D')

    da = DictArray(Dict)
    assert da._keys == auto_keys
    da.rename('x', 'a')
    assert da._keys == ('a', 'y', 'z')
    da.rename('y', 'b')
    assert da._keys == ('a', 'b', 'z')
    da.rename('z', 'c')
    assert da._keys == manual_keys
    da.rename('q', 'something')
    assert da._keys == (*manual_keys, 'something')
    assert 'q' in da._rename_dict.keys()

    with pytest.raises(ValueError):
        da.rename('x', 'a')


def test_construct_equal():
    da1 = DictArray()
    add_point_tester(da1)
    da2 = DictArray(da1)
    check_equal(da1, da2)


def test_append():
    logging.debug('---------------Begin test_append()')

    da = DictArray()
    add_point_tester(da)
    check_post_add_point(da)

    da = DictArray()
    da.append(1, 2)
    da.append({'x':3, 'y':4})
    da.append((5, 6))
    da.append([7, 8])
    da.append(y=10, x=9)
    da.append(np.array([11, 12]))

    assert np.all(da['x'] == [1,3,5,7,9,11])
    assert np.all(da['y'] == [2,4,6,8,10,12])
    assert np.all(da[:,0] == [1,3,5,7,9,11])
    assert np.all(da[:,'x'] == [1,3,5,7,9,11])
    assert np.all(da[:,1] == [2,4,6,8,10,12])
    assert np.all(da[:,'y'] == [2,4,6,8,10,12])
    assert np.all(da[0] == [1,2])
    assert np.all(da[1] == [3,4])
    assert np.all(da[2] == [5,6])
    assert np.all(da[3] == [7,8])
    assert np.all(da[4] == [9,10])
    assert np.all(da[5] == [11,12])

    # appending multiple points
    da = DictArray()
    da.append([1, 2], [3, 4])
    da.append({'x':[5, 7], 'y':[6, 8]})
    da.append(y=[10, 12], x=[9, 11])
    da.append(np.array([[13, 14], [15, 16]]))

    assert np.all(da['x'] == [1,3,5,7,9,11,13,15])
    assert np.all(da['y'] == [2,4,6,8,10,12,14,16])
    assert np.all(da[:,0] == [1,3,5,7,9,11,13,15])
    assert np.all(da[:,'x'] == [1,3,5,7,9,11,13,15])
    assert np.all(da[:,1] == [2,4,6,8,10,12,14,16])
    assert np.all(da[:,'y'] == [2,4,6,8,10,12,14,16])
    assert np.all(da[0] == [1,2])
    assert np.all(da[1] == [3,4])
    assert np.all(da[2] == [5,6])
    assert np.all(da[3] == [7,8])
    assert np.all(da[4] == [9,10])
    assert np.all(da[5] == [11,12])
    assert np.all(da[6] == [13,14])
    assert np.all(da[7] == [15,16])


def test_extend():
    logging.debug('---------------Begin test_extend()')

    da = DictArray()
    da.append(1, 2)
    da.extend(a=3)
    assert da.shape == (1,3)
    assert da._keys == ('x', 'y', 'a')
    assert da._rename_dict == OrderedDict(x='x', y='y', a='a')
    assert np.array_equal(da.array, [[1,2,3]])

    da = DictArray([[1,2]])
    da.extend(a=3, z=4)
    assert da.shape == (1,4)
    assert da._keys == ('x', 'y', 'a', 'z')
    assert da._rename_dict == OrderedDict(x='x', y='y', a='a', z='z')
    assert np.array_equal(da.array, [[1,2,3,4]])

    da = DictArray([[1,2]])
    da.extend([3,4])
    assert da.shape == (1,4)
    assert da._keys == ('x', 'y', 'z', 'a')
    assert da._rename_dict == OrderedDict(x='x', y='y', z='z', a='a')
    assert np.array_equal(da.array, [[1,2,3,4]])

    with pytest.raises(ValueError):
        # incorrect data count
        da.extend(b=[1,2])

    with pytest.raises(ValueError):
        # key already used
        da.extend(x=[1,2])


def test_append_extend():
    logging.debug('---------------Begin test_append_extend()')

    da = DictArray()
    da.append(1, 2)
    assert da.shape == (1,2)
    da.rename('z', 'a')
    da.append([4,5,6])
    assert da._keys == ('x', 'y', 'a')
    assert da._rename_dict == OrderedDict(x='x', y='y', z='a')
    assert np.allclose(da.array, [[1, 2, np.nan], [4, 5, 6]], equal_nan=True)

    da = DictArray([[1,2]])
    da.append([4,5,6])
    assert da._keys == ('x', 'y', 'z')
    da.rename('z', 'a')
    assert da._keys == ('x', 'y', 'a')
    assert da._rename_dict == OrderedDict(x='x', y='y', z='a')
    assert np.allclose(da.array, [[1,2,np.nan],[4,5,6]], equal_nan=True)
