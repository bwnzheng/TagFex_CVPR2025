class NOP(object):
    """
    Class that allows incomplete code to run without crashing
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return True

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        return self

    def __delitem__(self, key):
        return None

    def __iter__(self):
        return self

    def __reversed__(self):
        return self

    def __contains__(self, item):
        return False
        
    def __missing__(self, key):
        pass

    def __len__(self):
        return 0

    def __getattr__(self, attr):
        # print(f'accesssing nop by {attr}.')
        return self

    def __setattr__(self, name, value):
        return self

    def __delattr__(self, name):
        return None

    def __pos__(self):
        return self

    def __neg__(self):
        return self

    def __abs__(self):
        return self

    def __invert__(self):
        return self

    def __round__(self, n):
        return self

    def __floor__(self):
        return self

    def __ceil__(self):
        return self

    def __trunc__(self):
        return self

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __floordiv__(self, other):
        return self

    def __div__(self, other):
        return self

    def __truediv__(self, other):
        return self

    def __mod__(self, other):
        return self

    def __divmod__(self, other):
        return self

    def __pow__(self,n):
        return self

    def __lshift__(self, other):
        return self

    def __rshift__(self, other):
        return self

    def __and__(self, other):
        return self

    def __or__(self, other):
        return self

    def __xor__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __rsub__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __rfloordiv__(self, other):
        return self

    def __rdiv__(self, other):
        return self

    def __rtruediv__(self, other):
        return self

    def __rmod__(self, other):
        return self

    def __rdivmod__(self, other):
        return self

    def __rpow__(self, n):
        return self

    def __rlshift__(self, other):
        return self

    def __rrshift__(self, other):
        return self

    def __rand__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __rxor__(self, other):
        return self

    def __iadd__(self, other):
        return self

    def __isub__(self, other):
        return self

    def __imul__(self, other):
        return self

    def __ifloordiv__(self, other):
        return self

    def __idiv__(self, other):
        return self

    def __itruediv__(self, other):
        return self

    def __imod__(self, other):
        return self

    def __ipow__(self, n):
        return self

    def __ilshift__(self, other):
        return self

    def __irshift__(self, other):
        return self

    def __iand__(self, other):
        return self

    def __ior__(self, other):
        return self

    def __ixor__(self, other):
        return self

    def __int__(self):
        return self

    def __long__(self):
        return self

    def __float__(self):
        return self

    def __complex__(self):
        return self

    def __oct__(self):
        return self

    def __hex__(self):
        return self

    def __index__(self):
        return self

    def __trunc__(self):
        return self

    def __coerce__(self, other):
        return NotImplemented

    def __str__(self):
        return ''

    def __repr__(self):
        return 'NOP'

    def __unicode__(self):
        return u''

    def __bytes__(self):
        return b''

    def __format__(self, formatstr):
        return ''

    def __hash__(self):
        return None

    def __nonzero__(self):
        return False
    
    def next(self):
        raise StopIteration
        
    def __next__(self):
        raise StopIteration

