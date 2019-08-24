import pytest
import os, sys
import logging
import numpy as np
from datastream import ROI

TRACEBACK_FMT = 'File "%(pathname)s", line %(lineno)d:'
# logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.INFO, format='%(funcName)s:line %(lineno)d:%(message)s')
logging.basicConfig(stream=sys.stdout, filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')
# logging.basicConfig(filename='tests/logs/test_DictArray.log', filemode='w', level=logging.DEBUG, format=TRACEBACK_FMT+'%(message)s')

raw_data = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]


def test_constructor():
    logging.info('---------------Begin test_constructor()')
    data = np.array(raw_data)

    roi = ROI(data)
    assert np.array_equal(roi.base, data)
    assert np.array_equal(roi, data)
    assert roi.roi_shape == (3,4)
    assert roi.roi_offset == (0,0)

    roi = ROI(data, shape=(2))
    assert np.array_equal(roi, [[0,1,2,3],[4,5,6,7]])
    assert roi.roi_shape == (2,4)
    assert roi.roi_offset == (0,0)

    roi = ROI(data, shape=2)
    assert np.array_equal(roi, [[0,1,2,3],[4,5,6,7]])
    assert roi.roi_shape == (2,4)
    assert roi.roi_offset == (0,0)

    roi = ROI(data, shape=(2,2))
    assert np.array_equal(roi, [[0,1],[4,5]])
    assert roi.roi_shape == (2,2)
    assert roi.roi_offset == (0,0)

    roi = ROI(data, offset=(2))
    assert np.array_equal(roi, [[8,9,10,11]])
    assert roi.roi_shape == (3,4)
    assert roi.roi_offset == (2,0)

    roi = ROI(data, offset=2)
    assert np.array_equal(roi, [[8,9,10,11]])
    assert roi.roi_shape == (3,4)
    assert roi.roi_offset == (2,0)

    roi = ROI(data, offset=(2,2))
    assert np.array_equal(roi, [[10,11]])
    assert roi.roi_shape == (3,4)
    assert roi.roi_offset == (2,2)

    roi = ROI(data, shape=(2,1), offset=(1,2))
    assert np.array_equal(roi, [[6],[10]])
    assert roi.roi_shape == (2,1)
    assert roi.roi_offset == (1,2)

    assert np.array_equal(roi.base, raw_data)


def test_locate():
    logging.info('---------------Begin test_locate()')
    # [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    data = np.array(raw_data)

    roi = ROI(data).locate(2)
    assert np.array_equal(roi, [[8,9,10,11]])
    assert roi.roi_offset == (2,0)
    assert roi.roi_shape == (3,4)
    roi.locate((2,))
    assert np.array_equal(roi, [[8,9,10,11]])
    assert roi.roi_offset == (2,0)
    assert roi.roi_shape == (3,4)
    roi.locate((2,0))
    assert np.array_equal(roi, [[8,9,10,11]])
    assert roi.roi_offset == (2,0)
    assert roi.roi_shape == (3,4)

    roi.locate((0, 2))
    assert np.array_equal(roi, [[2,3],[6,7],[10,11]])

    roi.locate((2, 2))
    assert np.array_equal(roi, [[10,11]])

    roi = ROI(data, shape=(2, 2)).locate((1, 2))
    assert np.array_equal(roi, [[6,7],[10,11]])

    assert np.array_equal(roi.locate(None), [[0,1],[4,5]])
    assert np.array_equal(roi.base, raw_data)


def test_resize():
    logging.info('---------------Begin test_resize()')
    # [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    data = np.array(raw_data)

    roi = ROI(data).resize(2)
    assert np.array_equal(roi, [[0, 1, 2, 3], [4, 5, 6, 7]])
    roi = ROI(data).resize((2, ))
    assert np.array_equal(roi, [[0,1,2,3],[4,5,6,7]])

    roi = ROI(data).resize((2, 2))
    assert np.array_equal(roi, [[0,1],[4,5]])
    roi.resize((2, 3))
    assert np.array_equal(roi, [[0,1,2],[4,5,6]])
    roi.resize((3, 2))
    assert np.array_equal(roi, [[0,1],[4,5],[8,9]])

    roi = ROI(data, offset=(1, 1)).resize((2, 2))
    assert np.array_equal(roi, [[5,6],[9,10]])

    assert np.array_equal(roi.resize(None), [[5,6,7],[9,10,11]])
    assert np.array_equal(roi.base, raw_data)


def test_shift():
    logging.info('---------------Begin test_shift()')
    # [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    data = np.array(raw_data)

    roi = ROI(data).shift(1)
    assert np.array_equal(roi, [[4,5,6,7],[8,9,10,11]])
    roi = ROI(data).shift((1,))
    assert np.array_equal(roi, [[4,5,6,7],[8,9,10,11]])
    roi = ROI(data).shift((1,0))
    assert np.array_equal(roi, [[4,5,6,7],[8,9,10,11]])
    roi.shift((1, 2))
    assert np.array_equal(roi, [[10, 11]])

    roi.shift((0, -1))
    assert np.array_equal(roi, [[9, 10, 11]])

    roi.shift(-1)
    assert np.array_equal(roi, [[5, 6, 7],[9,10,11]])

    roi.shift((1, -1))
    assert np.array_equal(roi, [[8,9,10,11]])

    assert np.array_equal(roi.shift(None), raw_data)
    assert np.array_equal(roi.base, raw_data)


def test_stretch():
    logging.info('---------------Begin test_stretch()')
    # [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    data = np.array(raw_data)

    roi = ROI(data).stretch((-1,-2))
    assert np.array_equal(roi, [[0, 1], [4, 5]])
    assert np.array_equal(roi.stretch(1, 2), data)

    roi = ROI(data).stretch((1, 2))
    assert np.array_equal(roi, [[0,1,2,3],[4,5,6,7],[8,9,10,11]])
    assert np.array_equal(roi.stretch((1,2)), [[0,1,2,3],[4,5,6,7],[8,9,10,11]])

    assert np.array_equal(roi.stretch(-1,-2), [[0,1],[4,5]])
    assert np.array_equal(roi.stretch(0,-1), [[0],[4]])
    assert np.array_equal(roi.stretch(1,1), [[0,1],[4,5],[8,9]])

    assert np.array_equal(roi.stretch(None), raw_data)
    assert np.array_equal(roi.base, raw_data)


def test_set():
    logging.info('---------------Begin test_shift()')
    data = np.array(raw_data)

    roi = ROI(data, shape=(1,1), offset=(1,0))
    roi[:] = 0
    assert np.array_equal(roi, [[0]])
    assert np.array_equal(roi.base, [[0,1,2,3],[0,5,6,7],[8,9,10,11]])
    roi.shift(1,0)
    roi[0] = 0
    assert np.array_equal(roi, [[0]])
    assert np.array_equal(roi.base, [[0,1,2,3],[0,5,6,7],[0,9,10,11]])
    roi.shift(-2,1).resize(2,2)
    roi[:] = 0
    assert np.array_equal(roi, [[0,0],[0,0]])
    assert np.array_equal(roi.base, [[0,0,0,3],[0,0,0,7],[0,9,10,11]])


def test_limits():
    logging.info('---------------Begin test_limits()')
    # [[0,1],[2,3]]
    raw_data = [[0,1],[2,3]]
    data = np.array(raw_data)

    roi = ROI(data)
    assert np.array_equal(roi, data)
    assert np.array_equal(roi.stretch(2), [[0, 1], [2, 3]])
    assert np.array_equal(roi.stretch(-1), [[0, 1]])
    assert np.array_equal(roi.stretch(-1), np.ndarray(shape=(0,2)))
    assert np.array_equal(roi.stretch(1), [[0, 1]])
    assert np.array_equal(roi.stretch(0), [[0, 1]])
    assert np.array_equal(roi.stretch(1), [[0, 1], [2, 3]])

    assert np.array_equal(roi.stretch(0, 1), [[0, 1], [2, 3]])
    assert np.array_equal(roi.stretch(0, 1), [[0, 1], [2, 3]])
    assert np.array_equal(roi.stretch(0, -1), [[0], [2]])
    assert np.array_equal(roi.stretch(0, -1), np.ndarray(shape=(2,0)))

    assert np.array_equal(roi.stretch(None), raw_data)
    assert np.array_equal(roi.shift(-1), [[0,1]])
    assert np.array_equal(roi.shift(-1), np.ndarray(shape=(0,2)))
    assert np.array_equal(roi.shift(1), [[0,1]])
    assert np.array_equal(roi.shift(1), [[0,1],[2,3]])
    assert np.array_equal(roi.shift(1), [[2,3]])
    assert np.array_equal(roi.shift(1), np.ndarray(shape=(0,2)))

    assert np.array_equal(roi.shift(-1), [[2, 3]])
    assert np.array_equal(roi.shift(0,-1), [[2]])
    assert np.array_equal(roi.shift(0,1), [[2,3]])
    assert np.array_equal(roi.shift(-1,0), [[0,1],[2,3]])
    assert np.array_equal(roi.shift(0,-1), [[0],[2]])
    assert np.array_equal(roi.shift(0,-1), np.ndarray(shape=(2,0)))
    assert np.array_equal(roi.shift(0,2), [[0,1],[2,3]])
    assert np.array_equal(roi.shift(0,1), [[1],[3]])
    assert np.array_equal(roi.shift(0,1), np.ndarray(shape=(2,0)))

