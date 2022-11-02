from lib import *

### Functions for lab assignment

# Quantization for entire model
def quantize_model(model, num_bits):
    first = True
    w_max = 0
    w_min = 0
    for k, v in model.state_dict().items():
        if "weight" in k:
            if first:
                first = False
                w_max = torch.max(v).item()
                w_min = torch.min(v).item()
            else:
                w_max = max(w_max, torch.max(v).item())
                w_min = min(w_min, torch.min(v).item())
    max_int = (1 << (num_bits-1))-1
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in sd.items():
            if "weight" in k:
                v = ((v / w_max) * max_int).int()
                sd[k] = v

    model.load_state_dict(sd)
    return model

# Quantization per layer
def quantize_model_per_layer(model, num_bits, round=False):
    saved_clamps = {
        # Using int8 and some training for fine-tuning, we can get very solid test accuracy, 88-89%
        8: {'conv1.weight': 0.37297823071479796,
            'bn1.weight': 1.0899368524551392,
            'conv2.weight': 0.161872079372406,
            'bn2.weight': 1.3505200052261352,
            'lin2.weight': 0.21466276049613953,
            'conv1.bias': 0.3321022689342499,
            'bn1.bias': 0.005840758085250768,
            'conv2.bias': 0.002631304860115053,
            'bn2.bias': 0.0036059844493865935,
            'lin2.bias': 0.0098433993011713}
    }
    sd = model.state_dict()
    max_int = (1 << (num_bits-1))-1
    for k, v in sd.items():
        if "weight" in k or 'bias' in k:
            # if round:
            #     clamp_pt = max_int
            #     # sd[k] = torch.clamp(sd[k], min=-clamp_pt, max=clamp_pt)
            #     # sd[k] = ((v / clamp_pt) * max_int).int()
            #     sd[k] = torch.round(sd[k]).type(torch.int8)
            #     continue
            # elif num_bits in saved_clamps and k in saved_clamps[num_bits]:
            #     clamp_pt = saved_clamps[num_bits][k]
            #     sd[k] = torch.clamp(sd[k], min=-clamp_pt, max=clamp_pt)
            #     sd[k] = ((v / clamp_pt) * max_int).int()
            #     sd[k] = sd[k].type(torch.int8)
            #     model.load_state_dict(sd)
            #     continue

            w_max = torch.max(v).item()
            w_min = torch.min(v).item()
            w_max = max(abs(w_max), abs(w_min))
            prev_model = deepcopy(model)
            prev_sd = prev_model.state_dict()
            prev_sd[k] = ((v / w_max) * max_int).int()
            prev_sd[k] = prev_sd[k].type(torch.int8)
            prev_model.load_state_dict(prev_sd)
            clamp_pt = w_max
            clamp_dt = 0.01
            # Should probably use a special subset of training data for this...
            print('Testing clamp point ' + str(clamp_pt))
            prev_acc = test_model(prev_model)
            
            while True:
                clamp_pt = clamp_pt - clamp_dt#max(clamp_pt - clamp_dt, 0)
                curr_model = deepcopy(model)
                curr_sd = curr_model.state_dict()
                curr_sd[k] = torch.clamp(curr_sd[k], min=-clamp_pt, max=clamp_pt)
                curr_sd[k] = ((v / clamp_pt) * max_int).int()
                curr_sd[k] = curr_sd[k].type(torch.int8)
                curr_model.load_state_dict(curr_sd)
                print('Testing clamp point ' + str(clamp_pt))
                curr_acc = test_model(curr_model)

                if curr_acc < prev_acc or clamp_pt < 0:
                    print('Clamp point for layer ' + k + ': ' + str(clamp_pt+clamp_dt))
                    break

                prev_model = curr_model
                prev_acc = curr_acc
                prev_sd = curr_sd
            sd[k] = prev_sd[k]
            model.load_state_dict(sd)

    return model


def create_quantized_fused():
    # Load baseline model
    model = load_model_base()
    test_model(model)
    # Setup quantization
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.ao.quantization.default_qconfig
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1'], ['conv2', 'bn2']])
    # Prepare model for quantization
    torch.ao.quantization.prepare(model, inplace=True)
    # Train the model a little for quantization
    model = train_model(model, 0.001, 1)
    # Test the pruned model
    test_model(model)
    # Quantize the pruned model
    torch.ao.quantization.convert(model, inplace=True)
    # Test the quantized and pruned model
    test_model(model)
    return model

def create_hooks(model):
    def log_data(self, input, output):
        print('Input:')
        print(input[0].int_repr())
        print('Output:')
        print(output.int_repr())
        print('=====================================')
    model.pool1.register_forward_hook(log_data)
    model.conv2.register_forward_hook(log_data)

# Takes in a non-quantized model and tries to block out channels/filters to find pruning candidates
def prune_model(model):
    sd = model.state_dict()
    acc = test_model(model)
    for k, v in sd.items():
        if 'weight' in k and 'conv' in k:
            print('Trying to prune ' + k)
            for i in range(sd[k].size()[0]):
                cur_model = deepcopy(model)
                cur_sd = cur_model.state_dict()
                temp = torch.clone(cur_sd[k])
                temp[i] = torch.zeros(temp[i].size())
                temp = temp.contiguous()
                cur_sd[k] = temp
                cur_model.load_state_dict(cur_sd)
                print('Trying to prune ' + str(i))
                cur_acc = test_model(cur_model)

                if acc - cur_acc <= 1.0:
                    print('Prune candidate ' + k + '[' + str(i) + '] with cost ' + str(acc - cur_acc))
                    # sd[k][i] = torch.zeros(sd[k][i].size())
                    # model.load_state_dict(sd)
                    # acc = cur_acc
                    # print('Pruning ' + str(i) + ', updated accuracy: ' + str(acc))
    return model


# Using manually implemented quantization, try to quantize from bit representations 8 to 16
def step1A():
    model = load_model_base()
    x = []
    y = []
    for b in range(8, 17):
        model = load_model_base()
        model = quantize_model(model, b)
        acc = test_model(model)
        x.append(b)
        y.append(acc)

    plt.plot(x, y)
    plt.xlabel('Weight quantization target bits')
    plt.ylabel('Test Accuracy %')
    plt.show()

def step1B():
    def hist(model):
        arr = np.array([])
        sd = model.state_dict()
        for k, v in sd.items():
            if "weight" in k:
                arr = np.concatenate((torch.flatten(sd[k].data).numpy(), arr))
        return arr

    model = load_model_base()
    qmodel = quantize_model(deepcopy(model), 8)

    fig, axs = plt.subplots(1,2)
    axs[0].hist(hist(model), bins='auto')
    axs[0].set_title("Histogram of original weights")
    axs[1].hist(hist(qmodel), bins='auto')
    axs[1].set_title("Histogram of 8-bit quantized weights")
    plt.show()

    # Uncomment for seperate plots
    # plt.hist(hist(model), bins='auto')
    # plt.title("Histogram of original weights")
    # plt.show()
    
    # plt.hist(hist(qmodel), bins='auto')
    # plt.title("Histogram of 8-bit quantized weights")
    # plt.show()

    # Clamp model
    sd = model.state_dict()
    for k, v in sd.items():
        if "weight" in k:
            sd[k] = torch.clamp(sd[k], min=-0.29, max=0.29)
    model.load_state_dict(sd)

    # Quantize model
    qmodel_clamp = quantize_model(model, 8)
    test_model(qmodel)
    test_model(qmodel_clamp)

def step1C():
    model = load_model_base()
    # Quantize model per layer
    model = quantize_model_per_layer(model, 8)
    # Fine-tune train the model
    model = train_model(model, 0.01, 1)
    test_model(model)

def step2():
    def prune_layer(layer='conv1', ptype='filter'):
        dim = -1
        # Conv1 filter and channel pruning are the same since input channels is 1
        if layer == 'conv1':
            dim = 0
        elif layer == 'conv2':
            if ptype == 'filter':
                dim = 1
            elif ptype == 'channel':
                dim = 0

        num_samples = 25
        filter_prune_acc = []
        proportion = np.arange(0, 1.0, 1.0/num_samples)
        for i in proportion:
            model = load_model_base()
            print(i)
            model = prune_model_conv(model, layer, i, dim=dim)
            filter_prune_acc.append(test_model(model))

        plt.plot(proportion, filter_prune_acc)
        plt.title(layer + ' ' + ptype + ' pruning')
        plt.xlabel('prune proportion')
        plt.ylabel('Test Accuracy %')
        plt.show()

    def prune_conv(ptype='filter'):
        conv2_dim = -1
        if ptype == 'filter':
            conv2_dim = 1
        elif ptype == 'channel':
            conv2_dim = 0
        # Filter pruning cross of conv1 and conv2 at different proportions
        num_samples = 15
        max_proportion = 0.3
        filter_prune_acc = np.zeros((num_samples, num_samples))
        proportion_conv1 = np.arange(0, max_proportion, max_proportion/num_samples)
        proportion_conv2 = np.arange(0, max_proportion, max_proportion/num_samples)
        idx = 0
        for i in proportion_conv1:
            jdx = 0
            for j in proportion_conv2:
                model = load_model_base()
                print(str(i) + ', ' + str(j))
                model = prune_model_conv(model, 'conv1', i, dim=0)
                model = prune_model_conv(model, 'conv2', j, dim=conv2_dim)
                filter_prune_acc[idx][jdx] = test_model(model)
                jdx += 1
            idx += 1

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(proportion_conv1, proportion_conv2)
        surf = ax.plot_surface(X, Y, filter_prune_acc,
                                cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.title('conv1 and conv2 ' + ptype + ' pruning')
        ax.set_xlabel('conv1 prune proportion')
        ax.set_ylabel('conv2 prune proportion')
        ax.set_zlabel('Test Accuracy %')
        plt.show()


    def cust_prune_quant():
        model = load_model_base()
        model = quantize_model(model, 8)
        test_model(model)
        # Using channel pruning since our previous studies showed it had better behavior
        prune_model_conv(model, 'conv2', 0.1, dim=0)
        test_model(model)
        # We get low accuracies with custom quantization since we can't fine-tune
        # but this should be good enough and explain in report

    # prune_layer('conv1', 'filter')
    # prune_layer('conv2', 'filter')
    # prune_conv('filter')
    # prune_layer('conv2', 'channel')
    # prune_conv('channel')
    cust_prune_quant()

if __name__ == "__main__":
    # step1A()
    # step1B()
    # step1C()
    step2()