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

    from . import empty, dictOfLists, dictOfValues, listOfDicts, \
        listOfLists, listOfValues, ndarray, recarray, VALUES, \
        NAMEDTUPLEKEYS, RECARRAYKEYS, DICTKEYS

    KEYS = ()

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
