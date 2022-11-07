from lib import *
from lab import *


# Function for clamping weights of a given layer to a given floating point format
def clamp_weights(layer, sb, eb, mb):
    np_weights = layer.weight.detach().numpy()
    # WARNING: this function is *super* slow
    np_weights = vec_clamp_float(np_weights, sb, eb, mb)
    with torch.no_grad():
        layer.weight = nn.Parameter(torch.from_numpy(np_weights))

# Hook function that can be applied by attaching to the various model layers
def clamp_weights_hook(self, input, output):
    s_bits = clamp_weights_hook.s_bits
    e_bits = clamp_weights_hook.e_bits
    m_bits = clamp_weights_hook.m_bits
    # First, detach and convert the weights to a NumPy array
    np_weights = self.weight.detach().numpy()
    # Call the vectorized clamp function
    np_weights = vec_clamp_float(np_weights, s_bits, e_bits, m_bits)
    with torch.no_grad():
        self.weight = nn.Parameter(torch.from_numpy(np_weights))

# Example of attaching hooks
# model.conv1.register_forward_hook(clamp_weights_hook)
# model.bn1.register_forward_hook(clamp_weights_hook)
# model.conv2.register_forward_hook(clamp_weights_hook)
# model.bn2.register_forward_hook(clamp_weights_hook)
# model.lin2.register_forward_hook(clamp_weights_hook)


#############################
# Weight data sweep example #
#############################
man_bits = np.arange(0, 24, 1)
exp_bits = np.arange(0, 9, 1)
clamp_acc = np.zeros((24, 9))

for mb in man_bits:
    for eb in exp_bits:
        print(str(eb) + ', ' + str(mb))
        if eb == 0:
            clamp_acc[mb][eb] = 0
        else:
            model = load_model_base()
            # Rather than attaching weights, since we're only doing inference
            # We can just clamp the weights rather than using hooks that apply on every pass
            clamp_weights(model.conv1, 1, eb, mb)
            clamp_weights(model.bn1, 1, eb, mb)
            clamp_weights(model.conv2, 1, eb, mb)
            clamp_weights(model.bn2, 1, eb, mb)
            clamp_weights(model.lin2, 1, eb, mb)
            clamp_acc[mb][eb] = test_model(model)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(exp_bits, man_bits)
surf = ax.plot_surface(X, Y, clamp_acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('Weights floating point downconverting')
ax.set_xlabel('num exponent bits')
ax.set_ylabel('num significand bits')
ax.set_zlabel('Test Accuracy %')
plt.show()



############################
# Input data sweep example #
############################

man_bits = np.arange(0, 24, 1)
exp_bits = np.arange(0, 9, 1)
clamp_acc = np.zeros((24, 9))

for mb in man_bits:
    for eb in exp_bits:
        print(str(eb) + ', ' + str(mb))
        if eb == 0:
            clamp_acc[mb][eb] = 0
        else:
            model = load_model_base()
            # Call test_model with the parameters for this format
            # The testing function will format all input data into the network
            clamp_acc[mb][eb] = test_model(model, 1, eb, mb)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(exp_bits, man_bits)
surf = ax.plot_surface(X, Y, clamp_acc, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title('Input data floating point downconverting')
ax.set_xlabel('num exponent bits')
ax.set_ylabel('num significand bits')
ax.set_zlabel('Test Accuracy %')
plt.show()