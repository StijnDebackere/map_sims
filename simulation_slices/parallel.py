import multiprocessing
from multiprocessing import freeze_support, Process
import os
import queue
import time
from traceback import format_exc


class SharedCounter(object):
    """ A synchronized shared counter.

    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.
    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n=0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n=1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class Queue(queue.Queue):
    """ A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """

    def __init__(self, maxsize=0):
        # super(Queue, self).__init__(maxsize=maxsize, ctx=multiprocessing.get_context())
        super(Queue, self).__init__(maxsize=maxsize)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put((args, kwargs))

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """ Remove all elements from the Queue. """
        while not self.empty():
            self.get()


class Worker(Process):
    """Worker takes input to task_fn from worker_in_q and puts the results
    on worker_out_q.

    :param task_fn: callable for :class:`parallel.task_queue.Worker`


    """

    def __init__(self, task_fn, worker_in_q, worker_out_q):
        super(Worker, self).__init__()
        self.task_fn = task_fn
        self.worker_in_q = worker_in_q
        self.worker_out_q = worker_out_q

    def run(self):
        while self.worker_in_q.qsize() > 0:
            args, kwargs = self.worker_in_q.get()
            try:
                result = self.task_fn(*args, **kwargs)
                self.worker_out_q.put((result, None))
            except Exception as e:
                e.stack_trace = format_exc()
                self.worker_out_q.put((None, e))