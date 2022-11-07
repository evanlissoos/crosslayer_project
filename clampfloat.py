import struct
import numpy as np
from cffi import FFI

# C code version of the clamp float function for speedup
ffi = FFI()
ffi.set_source("_clampfloat", """
float clamp_float(float value, unsigned s_bits, unsigned e_bits, unsigned m_bits) {
    unsigned * value_i_ptr = (unsigned *) &value;
    unsigned value_i = *value_i_ptr;
    unsigned res_i = 0;
    float res_f;
    float * res_f_ptr = (float *) &res_i;

    // Truncate the mantissa
    unsigned man_size_diff = 23 - m_bits;
    unsigned man = (value_i & 0x7FFFFF) >> man_size_diff;
    man <<= man_size_diff;
    res_i |= man;

    // Compute the effective exponent, then clamp to the representable range
    int eff_exp = (value_i >> 23 & 0xFF) - 127;
    int max_exp = (1 << (e_bits-1))-1;
    int res_exp = (max_exp > eff_exp) ? eff_exp : max_exp;
    res_exp = (-max_exp > eff_exp) ? -max_exp : eff_exp;
    res_exp += 127; // Add back bias
    res_i |= res_exp << 23;

    // Handle the sign
    unsigned sign = value_i & 0x80000000;
    if(s_bits == 0 && sign != 0) {
        // If unsigned, clamp negatives to zero
        res_i = 0;
    } else {
        // Otherwise, push the sign bit back in
        res_i |= sign;
    }
    res_f = *res_f_ptr;
    return res_f;   
}
""")
ffi.cdef("""float clamp_float(float, unsigned, unsigned, unsigned);""")
ffi.compile()
from _clampfloat import lib  # import the compiled library

# Function that clamps a F32 value to a representable value given floating point parameters
# This is the faster function that calls the compiled C library
# Prints: nothing
# Retunrs: a clamped floating point value
def clamp_float(value_f, s_bits=1, e_bits=8, m_bits=23):
    return np.single(lib.clamp_float(value_f, s_bits, e_bits, m_bits))



# Function that converts float to integer
# Prints: nothing
# Returns: an integer value that represents the binary float value
def float_to_int(value):
    [d] = struct.unpack(">L", struct.pack(">f", value))
    return d

# Function that converts integer to float
# Prints: nothing
# Returns: the floating point value represented by the integer's binary value
def int_to_float(value):
    [f] = struct.unpack(">f", struct.pack(">L", value))
    return f

# OLD Function that clamps a F32 value to a representable value given floating point parameters
# This is the slow Python implementation
# Prints: nothing
# Retunrs: a clamped floating point value
def clamp_float_python(value_f, s_bits=1, e_bits=8, m_bits=23):
    # First, convert the float to an integer for bit manipulation
    value_i = float_to_int(value_f)
    res_i = 0

    # Truncate the mantissa
    man_size_diff = 23 - m_bits
    man = (value_i & 0x7FFFFF) >> man_size_diff
    man = man << man_size_diff
    res_i |= man

    # Compute the effective exponent, then clamp to the representable range
    eff_exp = (value_i >> 23 & 0xFF) - 127
    max_exp = (1 << (e_bits-1))-1
    res_exp = min(max_exp, eff_exp)
    res_exp = max(-max_exp, res_exp)
    res_exp = res_exp + 127 # Add back bias
    res_i |= res_exp << 23

    sign = value_i & 0x80000000
    # If unsigned, clamp negatives to zero
    if s_bits == 0 and sign != 0:
        res_i = 0
    # Otherwise, push the sign bit back in
    else:
        res_i |= sign

    # Convert the integer back to float and return
    res_f = int_to_float(res_i)
    return np.single(res_f)


# Create a NumPy vectorized version of this function
vec_clamp_float = np.vectorize(clamp_float)