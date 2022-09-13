import torch
import torch.nn as nn
from torchvision import datasets, models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_encoder(model_name, use_pretrained, feature_extract, output_size):
    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_size)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


latent_size = 256# pc+1encoder1decoder
half_latent_size = int(latent_size / 2)
encoder_hand, encoder_hand_input_size = get_encoder(
    'resnet', output_size=half_latent_size,
    use_pretrained=True, feature_extract=False)

encoder_input = torch.load("data/encoder_input.pt")
k = encoder_hand(encoder_input)
print(k.shape)
torch.onnx.export(encoder_hand, (encoder_input, ), "encoder_hand.onnx",
    input_names = ['encoder_input'],
    output_names = ["output"]
)
