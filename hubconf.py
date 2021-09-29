# import the necessary packages
import torch
from pyimagesearch import mlp

# define entry point/callable function 
# to initialize and return model
def custom_model():
    """ # This docstring shows up in hub.help()
    Initializes the MLP model instance
    Loads weights from path and
    returns the model
    """
	 model = mlp.get_training_model()
	model.load_state_dict(torch.load("model_wt.pth"))
	return model