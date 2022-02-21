import torch
import numpy as np

def show(data):
    print("#####")
    print(data[0, 0, :5])
    print(data[0, 1, :5])
    print(data[127, 126, :5])
    print(data[127, 127, :5])



a = torch.load("sdf_values_hand.pt")
show(a)
a = a.numpy()
#print(np.count_nonzero(a >= 0.0))
hand_vol = np.count_nonzero(a >= 0.0)
#print(a.)


b = torch.load("sdf_values_obj.pt")
show(b)
b = b.numpy()
#print(np.count_nonzero(b >= 0.0))
#print(np.count_nonzero(b < 0.0))
obj_vol = np.count_nonzero(b >= 0.0)

print(hand_vol / obj_vol)
