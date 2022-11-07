from model import *

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
    model.eval()
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
            prune.remove(module, 'weight')
    return model


# Function that saves a model to a file for PyTorch
# Prints: nothing
# Returns: nothing
def save_model(model, path):
    torch.save(model.state_dict(), path)

