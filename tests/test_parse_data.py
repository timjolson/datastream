import pytest
import os, sys
import logging
from collections import namedtuple
import numpy as np
from datastream import data_type, parse_data

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(name)s:%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')

# test data
dict1 = {'x': [], 'y': []}
dict2 = {'x': 0.0, 'y': 1.0}
dict3 = {'x': [0.0, 1.0], 'y': [2.0, 3.0]}
point_test = namedtuple('point_test', 'x, y')
ntup1 = point_test([], [])
ntup2 = point_test(0.0, 1.0)
ntup3 = point_test([0.0, 1.0], [2.0, 3.0])
tup1 = ([], [])
tup2 = (0.0, 1.0)
tup3 = ([0.0, 2.0], [1.0, 3.0])
nptup1 = (np.array([]), np.array([]))
nptup2 = (np.array(1.0), np.array(1.0))
nptup3 = (np.array([0.0, 2.0]), np.array([1.0, 3.0]))
np1 = np.array([[],[]])
np2 = np.array([0.0, 1.0])
np3 = np.array([[0.,2.],[1.,3.]])
list1 = [[], []]
list2 = [0.0, 1.0]
list3 = [[0.0, 2.0], [1.0, 3.0]]
sarr1 = np.array(list1)
sarr2 = np.array(list2)
sarr3 = np.array(list3)

VALUES1 = np.array([[],[]])
VALUES2 = np.array([[0.0],[1.0]])
VALUES3 = np.array([[0.0,2.0],[1.0,3.0]])
KEYS = ('x', 'y')


def test_dataType():
    logging.info('---------------Begin test_dataType()')

    assert data_type(dict1) == 'dictOfLists'
    assert data_type(dict2) == 'dictOfValues'
    assert data_type(dict3) == 'dictOfLists'

    assert data_type(ntup1) == 'listOfLists'
    assert data_type(ntup2) == 'listOfValues'
    assert data_type(ntup3) == 'listOfLists'

    assert data_type(tup1) == 'listOfLists'
    assert data_type(tup2) == 'listOfValues'
    assert data_type(tup3) == 'listOfLists'

    assert data_type(list1) == 'listOfLists'
    assert data_type(list2) == 'listOfValues'
    assert data_type(list3) == 'listOfLists'

    assert data_type(nptup1) == 'listOfLists'
    assert data_type(nptup2) == 'listOfLists'
    assert data_type(nptup3) == 'listOfLists'


def test_parse_data():
    logging.info('---------------Begin test_parse_data()')

    assert (parse_data(dict1) == VALUES1, KEYS)
    assert (parse_data(dict2) == VALUES2, KEYS)
    assert (parse_data(dict3) == VALUES3, KEYS)

    assert (parse_data(ntup1) == VALUES1, KEYS)
    assert (parse_data(ntup2) == VALUES2, KEYS)
    assert (parse_data(ntup3) == VALUES3, KEYS)

    assert (parse_data(tup1) == VALUES1, KEYS)
    assert (parse_data(tup2) == VALUES2, KEYS)
    assert (parse_data(tup3) == VALUES3, KEYS)

    assert (parse_data(nptup1) == VALUES1, KEYS)
    assert (parse_data(nptup2) == VALUES2, KEYS)
    assert (parse_data(nptup3) == VALUES3, KEYS)

    assert (parse_data(list1) == VALUES1, KEYS)
    assert (parse_data(list2) == VALUES2, KEYS)
    assert (parse_data(list3) == VALUES3, KEYS)

    assert (parse_data(nptup1) == VALUES1, KEYS)
    assert (parse_data(nptup2) == VALUES2, KEYS)
    assert (parse_data(nptup3) == VALUES3, KEYS)

    assert (parse_data(sarr1) == VALUES1, KEYS)
    assert (parse_data(sarr2) == VALUES2, KEYS)
    assert (parse_data(sarr3) == VALUES3, KEYS)

