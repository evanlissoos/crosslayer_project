from lib import *
from lab import *

model = load_model_base()
print(model.conv1.weight)
# First, detach and convert the weights to a NumPy array
np_weights = model.conv1.weight.detach().numpy()
# Call the vectorized clamp function
clamped_weights = vec_clamp_float(np_weights, 1, 8, 1)
# Assign the weights to the clamped weights
with torch.no_grad():
    model.conv1.weight = nn.Parameter(torch.from_numpy(clamped_weights))
print(model.conv1.weight)