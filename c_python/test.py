"""
Define the C-variables and functions from the C-files that are needed in Python
"""
from ctypes import c_float, c_int, c_uint, CDLL
import sys

lib_path = './ccode_lib_%s.so' % (sys.platform)
try:
    basic_function_lib = CDLL(lib_path)
except:
    print('OS %s not recognized' % (sys.platform))

python_c_square = basic_function_lib.c_square
python_c_square.restype = None

def do_square_using_c(tensor):
    n = 10#torch.numel(tensor)
    print(tensor.data_ptr())
    c_arr_in = (c_uint * n)(tensor.data_ptr())
    # c_arr_in = (c_double * n)(*list_in)
    # c_arr_out = (c_double * n)()

    python_c_square(c_uint(n), c_arr_in)

do_square_using_c(t)
print(t)
quit()
