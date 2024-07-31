import yaml

with open('ptq-adaround.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    
import code
code.interact(local=locals())