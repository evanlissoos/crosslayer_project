from model import *
import struct

# Function that loads the baseline model (FP32)
# Prints: nothing
# Returns: loaded model
def load_model_base():
    fname = './models/baseline.model'
    model = MyConvNet(args)
    model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    return model


# Function that loads the quantized model (int8)
# Prints: nothing
# Returns: loaded model
def load_model_quant():
    # https://discuss.pytorch.org/t/how-to-load-quantized-model-for-inference/140283
    fname = './models/qint8.model'
    # Start with the baseline
    model = load_model_base()
    # Quantize/convert the baseline model
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.ao.quantization.default_qconfig
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    # Now load in the saved quantized weights
    model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    return model

# Function that loads the quantized conv/bn fused model (int8)
# Prints: nothing
# Returns: loaded model
def load_model_fquant():
    # https://discuss.pytorch.org/t/how-to-load-quantized-model-for-inference/140283
    fname = './models/qint8_fused.model'
    # Start with the baseline
    model = load_model_base()
    # Not sure why, but need to test model otherwise it'll think we're trying to train and fail an assert
    # Probably some sort of PyTorch bug on load-to-use
    test_model(model, False)
    # Quantize/convert the baseline model
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.ao.quantization.default_qconfig
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1'], ['conv2', 'bn2']])
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    # Now load in the saved quantized weights
    model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    return model


# Function that calculates the size of the model
# Prints: size of the model (in MB)
# Returns: size of the model (in MB)
def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size_mb = os.path.getsize("temp.p")/1e6
    print('Size (MB):', size_mb)
    os.remove('temp.p')
    return size_mb


# Function that quantizes and fine-tunes a model using PyTorch functionality
# Prints: training and testing information
# Returns: quantized model
def torch_quant(model):
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.ao.quantization.default_qconfig
    torch.ao.quantization.prepare(model, inplace=True)
    model = train_model(model, 0.001, 2)
    torch.ao.quantization.convert(model, inplace=True)
    test_model(model)
    return model


# Function that performs l1 structured pruning across dimension 0 (convolution filters)
# Prints: nothing
# Returns: pruned model
# dim=0: channel pruning
# dim=1: filter pruning
def prune_model_conv(model, layer='conv', proportion=0.5, dim=0):
    for name, module in model.named_modules():
        if layer in name:
            prune.ln_structured(module, 'weight', proportion, dim=dim, n=1)
            # prune.remove(module, 'weight')
    return model


# Function that saves a model to a file for PyTorch
# Prints: nothing
# Returns: nothing
def save_model(model, path):
    torch.save(model.state_dict(), path)


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


# Function that clamps a F32 value to a representable value given floating point parameters
# Prints: nothing
# Retunrs: a clamped floating point value
def clamp_float(value_f, s_bits=1, e_bits=8, m_bits=23):
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