from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2

from Layers.adapter import adapter
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

model = adapter()

get_parameter_number(model.feedforward_adapter(input_tensor=(384,1)))

print(get_parameter_number(model.feedforward_adapter(input_tensor=(384,1))))