import pytest
import os, sys
import logging
from collections import namedtuple
import numpy as np
from datastream import data_type, parse_data

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(name)s:%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')


def test_dataType():
    logging.info('---------------Begin test_dataType()')

    # test data
    KEYS = None  # default keys, when none are specified
    DICTKEYS = ('a', 'b')  # keys for dict results (have `dict` in test case name)
    RECARRAYKEYS = ('c', 'd')  # keys for structured ndarray results (have `rec` in test case name)
    NAMEDTUPLEKEYS = ('Y', 'Z')  # keys for namedtuple results  (have `ntup` in test case name)

    VALUES = {  # test result values (append a VALUES key to end of each test case name)
        '0': np.array([]),
        '1': np.array([[], []]),  # (2,0)
        '2': np.array([[0.0, 1.0]]),  # (1,2)
        '3': np.array([[0.0, 1.0], [2.0, 3.0]]),  # (2,2)
        '4': np.ndarray((0,2))  # (0,2) empty
    }

    point_test = namedtuple('point_test', 'Y, Z')  # test namedtuple

    list1 = [[], []]  # test lists
    list2 = [0.0, 1.0]
    list3 = [[0.0, 1.0], [2.0, 3.0]]  # first point: x=0, y=1; second point: x=2, y=3

    # groups of test cases (group name matches data_type() output)
    empty = dict(empty0_0=dict(), empty1_0=(), empty2_0=[])
    dictOfLists = dict(
        dict4={'a': [], 'b': []},
        dict3={'a': [0.0, 2.0], 'b': [1.0, 3.0]}
    )
    dictOfValues = dict(dict2={'a': 0.0, 'b': 1.0})
    listOfLists = dict(
        list1=list1,
        list3=list3,
        ntup3=point_test([0.0, 2.0], [1.0, 3.0]),
        ntup4=point_test([], []),
        tup1=tuple(list1),
        tup3=tuple(list3),
        nptup1=(np.array([]), np.array([])),
        nptup3=(np.array([0.0, 1.0]), np.array([2.0, 3.0])),
        ntuplist3=[point_test(0., 1.), point_test(2.,3.)],
    )
    listOfValues = dict(
        list2=list2,
        ntup2=point_test(*list2),
        tup2=tuple(list2),
        np2=np.array(list2),
        arr2=np.array(tuple(list2)),
        nptup2=(np.array(0.0), np.array(1.0))  # single element arrays are considered values
    )
    listOfDicts = dict(
        lod2=[{'a': 0.0, 'b': 1.0}],
        lod3=[{'a': 0.0, 'b': 1.0}, {'a': 2.0, 'b': 3.0}]
    )
    ndarray = dict(
        np1=np.array(list1),
        np3=np.array(list3),
    )
    recarray = dict(
        recarr4=np.array([], dtype=[('c', float), ('d', float)]),
        recarre_4=np.array([[]], dtype=[('c', float), ('d', float)]),
        recarr1_4=np.array([[],[]], dtype=[('c', float), ('d', float)]),
        recarr2=np.array([(0.0, 1.0)], dtype=[('c', float), ('d', float)]),
        recarr3=np.array([(0.0, 1.0), (2.0, 3.0)], dtype=[('c', float), ('d', float)]),
    )

    for group in ['empty', 'dictOfLists', 'dictOfValues', 'listOfDicts', 'listOfLists', 'listOfValues', 'ndarray', 'recarray']:
        for k,v in locals()[group].items():
            logging.info(f"Testing {group} : {k} : {repr(v)}")
            assert data_type(v) == group

            vcheck = VALUES[k[-1]]
            if group.lower().find('dict')!=-1:
                kcheck = DICTKEYS
            elif group.lower().find('rec')!=-1:
                kcheck = RECARRAYKEYS
            elif k.lower().find('ntup')!=-1:
                kcheck = NAMEDTUPLEKEYS
            else:
                kcheck = KEYS
            logging.info(f"kcheck={kcheck}, vcheck={repr(vcheck)}")
            result = parse_data(v)
            logging.info(f"result={result}")
            assert np.array_equal(result[0], vcheck)
            assert result[1] == kcheck

    assert data_type(None) is None
