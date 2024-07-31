import torchvision.models as models                           # for example model
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy
from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
from mqbench.advanced_ptq import ptq_reconstruction
import yaml, dotmap, torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.models import *
from src.utils import *
from src.options import *
args = parse_arguments()

device = 'cuda'

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
    config = dotmap.DotMap(config)

model = eval(config.model.type)()
if config.model.path is not None:
    model.load_state_dict(torch.load(config.model.path))
    print("load model from: ", config.model.path)
# model = models.__dict__[config.model.type](pretrained=True)     # use vision pre-defined model
model.eval()

backend = eval(f"BackendType.{config.quantize.backend}")
extra_config = config.extra_config
model = prepare_by_platform(model, backend, extra_config)           # trace model and add quant nodes for model on backend       

""" load data """
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root=config.data.path, train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root=config.data.path, train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.data.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.data.batch_size, shuffle=False)
acc_before_quant = evaluate(val_loader=testloader, model=model, device=device)

# Set the data loaders as the input data for the model
cali_data = load_calibrate_data(trainloader, cali_batchsize=config.quantize.cali_batchsize)
model.eval()

with torch.no_grad():
    model.to(device)
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    for batch in cali_data:
        model(batch.to(device))
    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    # for batch in cali_data:
        # model(batch.to(device))
    model(cali_data[0].to(device))

if hasattr(config.quantize, 'reconstruction'):
    model = ptq_reconstruction(model, cali_data, config.quantize.reconstruction)

enable_quantization(model)

# """ evaluate """
acc_after_quant = evaluate(val_loader=testloader, model=model, device=device)

print("accuracy before quantization: ", acc_before_quant)
print("accuracy after quantization: ", acc_after_quant)

# """ export quantized model """
input_shape={'input': [1, 3, 32, 32]}
output_path = "./"
convert_deploy(model, backend, input_shape, 
               output_path=output_path, 
               model_name=config.output.filename)                   #! line 4. remove quant nodes, ready for deploying to real-world hardware



# if config.quantize.quantize_type == 'advanced_ptq':

# elif config.quantize.quantize_type == 'naive_ptq':