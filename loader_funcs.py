import numpy as np
import os
from datastream import DataStream

time, x, y = [0,1,2,3], [4,5,6,7], [8,9,10,11]
data_set_dir = os.path.join('tests', 'data_sets')
filenames = [os.path.join(data_set_dir,f) for f in os.listdir(data_set_dir) if not f.startswith('.')]

known = DataStream({'x':x,'y':y,'time':time})
for f in filenames:
    print(f)
    r = DataStream(f)
    if not f.endswith('empty'):
        try:
            assert np.array_equal(r, known)
        except AssertionError:
            print('############', f)
            print(known)
            print(r)
        if f.find('dict') != -1 or f.find('header') != -1 or f.find('save_struct') != -1:
            assert set(r.keys()) == set(known.keys())
        else:
            assert set(r.keys()) == set(['x', 'y', 'z'])
    else:
        assert r.size == 0
print('COMPLETED')