


import numpy as np



from functools import lru_cache, wraps
import numpy as np

# not used for the moment
def np_cache(*args, **kwargs):
    """
    LRU cache implementation for functions whose parameter at ``array_argument_index`` is a numpy array of dimensions <= 2
    
    https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75

    Example:
    >>> from sem_env.utils.cache import np_cache
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     return factor * array
    >>> multiply(array, 2)
    >>> multiply(array, 2)
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    
    def decorator(function):
        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            hashable_array = tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator



def L(p, x_n, x_m):
    return p * (np.sinh(x_n / p) - np.sinh(x_m / p))

def x_m(a, b, p):
    return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

def x_n(a, b, p):
    return x_m(a,b,p) + a

def z(x, p, x_m):
    # repeating value to perform multidim operation
    xx = x.T + x_m # >>> why + x_m?
    # self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
    # Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
    # pp = p[:, np.newaxis]

    rr = p * (np.cosh(xx / p) - 1)

    # reshaping back to p,x -> (vertical, horizontal)
    return rr.T

def T_moy(p, L, x_n, x_m, lineic_weight):
    
    a = x_n - x_m


    return p * lineic_weight * (a + (np.sinh(2*x_n / p) - np.sinh(2*x_m / p))*p/2) / L /2


# def z_from_x_2ddl(span):

#     span.compute()
#     span
#     return b * np.sinh(x / p) / np.sinh(a / (2 * p))



