# find the RF for PredNet
import numpy as np
import matplotlib.pyplot as plt
from border_ownership.prednet_rf_finder import compute_rf_mask

input_width = (128, 160)
query_neural_id = np.array([[2, 3]])
module_name = 'A3'

rf_mask = compute_rf_mask(module_name, query_neural_id)

plt.figure()
plt.imshow(rf_mask)
plt.show()
