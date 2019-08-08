# DataStream
Wrapper around `np.array` that can use keys to access the first axis. Also has convenience functions for data formatting and recording.

Similar in usage to a ``{'x':np.array(~), 'y':np.array(~)}`` or `np.recarray`:
without sacrificing numpy indexing/slicing features.

## DictArray

Wrapper around np.array that can use keys to access first axis.

    DictArray(data=None, **kwargs) -> np.array(~data, **kwargs)
        data: can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
            `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
        kwargs: passed to np.array() after processing `data`
        NOTE: any `list` in above formats can be a sequence with __iter__

    Methods:
    .append(*data, **kwargs): add data to the array, where `data` fits format
            above, OR keywords can be used
    .keys() ~= dict().keys()
    .items() ~= dict().items()
    .values() ~= dict().values()
    .as_dict() returns dict version of .items()
    
    CAUTION: keys can be any valid dict key, except the types
        tuple/slice/list/int/np.ndarray that have no custom handling.

    All np.array attributes are implicitly delegated to DictArray().array

#### Example Usage:
    obj = DictArray([[0,2,4],[1,3,5]])
    obj['x'] == obj[0] == obj[0,:] == [0,2,4]
    obj['x',1] == obj[0,1] == 2
    obj['y'] == obj[1] == obj[1,:] == [1,3,5]
    obj['y',1] == obj[1,1] == 3

    x, y = object(), object()
    obj = DictArray({x:[0], y:[1]})
    obj[x] == obj[0] == obj[0,:] == [0]
    obj[x,0] == obj[0,0] == 0
    obj[y] == obj[1] == obj[1,:] == [1]
    obj[y,0] == obj[1,0] == 1

## DataStream

DictArray with data recording functions.

    .set_record_file(file, format)
        set a file/object/function used to record data points.
        `file`:
            str -> filepath -> .write(msg)
            io.IOBase -> uses .write(msg)
            logging.Logger -> uses Logger.info(msg)
            logging.Handler -> uses Handler.handle(logging.getLogRecordFactory(msg))
            callable -> calls, passing the new data point, unformatted
        `format`: str: 'csv', 'dict', or 'list'

    .append  # extended to record new data when applicable

    .start_recording()  # begin recording appended data 
    .stop_recording()  # stop recording appended data
