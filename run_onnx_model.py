import torch
import onnxruntime as ort
import numpy as np

"""
graph torch-jit-export (
  %batch_imgs[FLOAT, 1x3x224x224]
  %template_3d_joints[FLOAT, 1x21x3]
  %template_vertices_sub[FLOAT, 1x195x3]
) initializers (
"""
encoder_input_obj = torch.load("data/encoder_input_obj.pt")
encoder_input = torch.load("data/encoder_input.pt")

#ort_sess = ort.InferenceSession('encoder_hand.onnx')
ort_sess = ort.InferenceSession('grasping_field.onnx')
ort_sess.get_modelmeta()
for input_item in ort_sess.get_inputs():
   print(f"Input: name={input_item.name}")
#first_output_name = ort_sess.get_outputs()[0].name
print(f"encoder_input: {encoder_input.shape}")
print(f"encoder_input_obj: {encoder_input_obj.shape}")
#input_names = ['encoder_input', 'encoder_input_obj'],
sdf_values_hand, sdf_values_obj, voxel_origin, voxel_size = ort_sess.run(None, {"encoder_input": encoder_input.numpy(), "encoder_input_obj": encoder_input_obj.numpy()})
# pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att, *rest = outputs
# # Print Result 
np.save('onnx_output_hand', sdf_values_hand)
print("saved.")
np.save('onnx_output_obj', sdf_values_obj)
print("saved.")
#print(outputs[2])
# print(pred_vertices.shape)
# print(pred_vertices)

