import torch
import numpy as np

def show(name, data):
    print(f"##### {name}:")
    print(data[0, 0, :5])
    print(data[0, 1, :5])
    print(data[127, 126, :5])
    print(data[127, 127, :5])


d = np.load('onnx_output_hand.npy')
d = torch.from_numpy(d)
show("hand", d)
d = np.load('onnx_output_obj.npy')
d = torch.from_numpy(d)
show("obj", d)
