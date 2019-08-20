import numpy as np
from collections import namedtuple

# test data
KEYS = ('x', 'y')  # default keys, when none are specified
DICTKEYS = ('a', 'b')  # keys for dict results (have `dict` in test case name)
RECARRAYKEYS = ('c', 'd')  # keys for structured ndarray results (have `rec` in test case name)
NAMEDTUPLEKEYS = ('Y', 'Z')  # keys for namedtuple results  (have `ntup` in test case name)

VALUES = {  # test result values (append a VALUES key to end of each test case name)
    '0': np.array([]),
    '1': np.array([[], []]),
    '2': np.array([[0.0, 1.0]]),
    '3': np.array([[0.0, 1.0], [2.0, 3.0]]),
    '4': np.ndarray((0,2))
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

groups = {k:v for k,v in locals().items() if isinstance(v, dict)}
list(map(groups.pop, (k for k,v in list(groups.items()) if k.startswith('_'))))
groups.pop('VALUES')
