import functools
import numpy as np


def lru_cache(maxsize=128, typed=False):
    """Taken from functools. Mods: skips caching unhashable inputs"""
    if maxsize is not None and not isinstance(maxsize, int):
        raise TypeError('Expected maxsize to be an integer or None')

    def decorating_function(user_function):
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, functools._CacheInfo)
        return functools.update_wrapper(wrapper, user_function)

    return decorating_function


def _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo):
    """Taken from functools. Mods: skips caching unhashable inputs"""
    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = functools._make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get    # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = functools.RLock()           # because linkedlist updates aren't threadsafe
    root = []                # root of the circular doubly linked list
    root[:] = [root, root, None, None]     # initialize by pointing to self

    if maxsize == 0:

        def wrapper(*args, **kwds):
            # No caching -- just a statistics update after a successful call
            nonlocal misses
            result = user_function(*args, **kwds)
            misses += 1
            return result

    elif maxsize is None:

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            try:
                hash((args, tuple(kwds.items())))
            except TypeError:
                misses += 1
                return user_function(*args, **kwds)
            else:
                key = make_key(args, kwds, typed)
                result = cache_get(key, sentinel)
                if result is not sentinel:
                    hits += 1
                    return result
                result = user_function(*args, **kwds)
                cache[key] = result
                misses += 1
                return result

    else:

        def wrapper(*args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full
            try:
                hash((args, tuple(kwds.items())))
            except TypeError as e:
                misses += 1
                return user_function(*args, **kwds)
            else:
                key = make_key(args, kwds, typed)
                with lock:
                    link = cache_get(key)
                    if link is not None:
                        # Move the link to the front of the circular queue
                        link_prev, link_next, _key, result = link
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        hits += 1
                        return result
                result = user_function(*args, **kwds)
                with lock:
                    if key in cache:
                        pass
                    elif full:
                        # Use the old root to store the new key and result.
                        oldroot = root
                        oldroot[KEY] = key
                        oldroot[RESULT] = result
                        # Empty the oldest link and make it the new root.
                        root = oldroot[NEXT]
                        oldkey = root[KEY]
                        oldresult = root[RESULT]
                        root[KEY] = root[RESULT] = None
                        # Now update the cache dictionary.
                        del cache[oldkey]
                        cache[key] = oldroot
                    else:
                        # Put result in a new link at the front of the queue.
                        last = root[PREV]
                        link = [last, root, key, result]
                        last[NEXT] = root[PREV] = cache[key] = link
                        full = (cache_len() >= maxsize)
                    misses += 1
                return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper


@lru_cache(512, typed=True)
def contains_non_index(key):
    if isinstance(key, (int, type(None), type(Ellipsis))):
        return False
    if hasattr(key, '__iter__'):
        if isinstance(key, str):
            return True
        return any(map(contains_non_index, key))
    elif isinstance(key, slice):
        return any(map(contains_non_index, (key.start, key.stop, key.step)))
    else:
        return True


def wrap_slice(self, start, stop, axis):
    b = self.base
    lim = b.shape[axis]
    # print('wrap_slice', start, stop, axis, lim)
    if start == stop:
        indices = slice(0,0)
    else:
        indices = tuple(map(lambda n: n%lim, range(start, stop)))
    # print('indices', indices)
    return indices


def fill_tuple(input, default):
    if not hasattr(input, '__iter__'):
        input = (input,)
    if not hasattr(default, '__iter__'):
        default = (default,)
    ls = len(input)
    lb = len(default)
    rv = []
    for i in range(lb):
        if i < ls:
            rv.append(input[i])
        else:
            rv.append(default[i])
    return tuple(rv)


def update_view(self):
    key = []
    shape = self._roi_shape
    offset = self._roi_offset
    for i in range(self.base.ndim):
        key.append(slice(*slice(max(offset[i], 0), max(offset[i]+shape[i], 0)).indices(self.base.shape[i])))
    key = tuple(key)
    # print('final key', key)
    rv = self.base[key]
    return rv


def do_keyed_slice(self, key):
    start, stop, step = key.start, key.stop, key.step
    if contains_non_index(step):
        raise TypeError("slice step must be an integer or None")
    step_shift = 1 if step is None else (1 if step > 0 else -1)
    start = process_getitem_key(self, start)
    if contains_non_index(stop):
        stop = process_getitem_key(self, stop)
    stop += step_shift
    if stop < 0:
        stop = None
    return slice(start, stop, step)


def process_getitem_key(self, key, axis=0):
    # TODO: add float handling
    if not contains_non_index(key):
        return key
    if isinstance(key, slice):
        return do_keyed_slice(self, key)
    elif isinstance(key, tuple):
        return tuple(process_getitem_key(self, k, axis) for k in key)
    elif isinstance(key, list):
        return list(process_getitem_key(self, k, axis) for k in key)
    else:
        try: return self._keys.index(key)
        except ValueError: raise KeyError(repr(key))


def is_sequence(obj):
    # taken from pyqtgraph.graphicsitems.PlotDataItem
    return hasattr(obj, '__iter__') or isinstance(obj, np.ndarray) or (
            hasattr(obj, 'implements') and obj.implements('MetaArray'))
