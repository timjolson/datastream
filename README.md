# DataStream
Wrapper around `np.array` that can use keys to access the first axis. Also has convenience functions for data formatting and recording.

Similar in usage to a ``{'x':np.array(~), 'y':np.array(~)}`` or `np.recarray`:
without sacrificing numpy indexing/slicing features.

## DictArray

Wrapper around `np.array` that can use keys to access the first axis.

    DictArray(data=None, **kwargs)
    `kwargs` are passed to np.array after processing `data`

    `data` can be any of `None`, `DictArray`, `dict of lists`, `dict of values`,
            `list of lists`, `list of dicts`, `list of values`, `np.recarray`, `np.ndarray`
    
    *note `list` here can be a sequence with `__iter__`

#### Methods

    .keys() ~= dict().keys()
    .items() ~= dict().items()
    .values() ~= dict().values()
    .as_dict() returns dict version of .items()
    
    np.array attributes are implicitly delegated to obj.points

#### Example Usage:
    obj = DictArray([0,1])
    obj['x'] == obj[0] == obj[0,:] == [0]
    obj['x',0] == obj[0,0] == 0
    obj['y'] == obj[1] == obj[1,:] == [1]
    obj['y',0] == obj[1,0] == 1

    obj = DictArray([[0,2,4],[1,3,5])
    obj['x'] == obj[0] == obj[0,:] == [0,2,4]
    obj['x',1] == obj[0,1] == 1
    obj['y'] == obj[1] == obj[1,:] == [1,3,5]
    obj['y',1] == obj[1,1] == 3

## DataStream

`DictArray` with data appending and recording functions.
    
    .add_points(*data, **kwargs)
        add data points to DataStream where `data` matches a format 
        supported by DictArray(), OR keywords can be used

    .set_record_file(file, format)
        set a file/object/function used to record data points.
        `file`:
            str -> filepath -> .write(msg)
            io.IOBase -> uses .write(msg)
            logging.Logger -> uses Logger.info(msg)
            logging.Handler -> uses Handler.handle(logging.getLogRecordFactory(msg))
            callable -> calls, passing the new data point, unformatted
        `format`: str: 'csv', 'dict', or 'list'

    .start_recording()  # begin recording appended data 
    .stop_recording()  # stop recording appended data
