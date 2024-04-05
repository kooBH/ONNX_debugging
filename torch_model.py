import torch
import torch.nn as nn
import torch.onnx

import onnx
import onnxruntime as ort

import numpy as np

class Model(nn.Module) :
    def __init__(self, n_channels = 10):
        super(Model,self).__init__()
        self.layer1 = nn.Conv2d(n_channels,n_channels,1)
        #self.act = nn.GroupNorm(n_channels,n_channels)
        self.act = nn.InstanceNorm2d(n_channels)
        self.layer2 = nn.Conv2d(n_channels,1,1)
        self.layer3 = nn.Linear(25,10)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = torch.reshape(x,(x.shape[0],25))
        x = self.layer3(x)
        
        return x
    
B = 1
C = 10
F = 5
T = 5

model = Model(n_channels = C)

optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.MSELoss()

for i in range(100) :
    x = torch.rand(B,C,F,T)
    gt = torch.rand(B,10)
    y = model(x)
    loss = criterion(y,gt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

## tracing
input = torch.rand(B,C,F,T)
torch_out = model(input.float())

#with open('input.bin', 'wb') as f:
np_in = input.float().detach().numpy()
np_in.tofile("input.bin")
#with open('output_python.bin', 'wb') as f:
np_out =  torch_out.float().detach().numpy()
np_out.tofile("output_python.bin")

torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL)

torch.onnx.export(
model,         # model being run 
input,       # model input (or a tuple for multiple inputs) 
"export.onnx",       # where to save the model  
opset_version=12,
do_constant_folding=False,
keep_initializers_as_inputs=False,
input_names = ['input'], 
output_names = ['output'],
#dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#'output' : {0 : 'batch_size'}},
export_params=True,
#verbose = True,
training = torch.onnx.TrainingMode.EVAL
)

#onnx_model = onnx.load("export.onnx")
#onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession(
 "export.onnx",
 # providers=["CUDAExecutionProvider"]
  providers=["CPUExecutionProvider"]
)
x = input.numpy()
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)

torch_out = torch_out.detach().numpy()

print(ort_outs)
print(torch_out)
np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-05, atol=1e-05)

print("PASS!!")