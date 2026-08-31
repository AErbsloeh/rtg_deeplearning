# Software for RTG Specialized Course - Deep Learning

## Introduction
This repo helps you to guide through the exercise to train deep neural networks in Python.
Here, we will use the *PyTorch* framework inside the *denspp.offline* framework. You find further information [here](https://github.com/es-ude/denspp.offline).
Software for RTG Specialized Course - Deep Learning

## Installation Guide
### Tools
To use this software, we recommend to install the following tools:
- uv package manager ([Link](https://docs.astral.sh/uv/) for installation)
- VScode ([Link](https://code.visualstudio.com) for Downloading and Installation)
- Git ([Link](https://git-scm.com/) for Downloading and Installation)

### Initialisation
To download the code open a terminal/cmd and navigate to the target directory for example: <br>
```cd C:\Users\Student\Documents\GitHub``` <br>
Then clone the repository (e.g., download a copy of the code): <br>
```git clone https://github.com/AErbsloeh/rtg_deeplearning``` <br>
This will automatically create a subfolder in your directory named rtg_deeplearning. Go to your code editor of choice (e.g. VScode) and navigate to this folder. <br>
To use the code, you have to initialise the project. Please run the following steps:
- Please enter ```uv sync``` into the terminal in order to install all packages.
- After it, please run the ```init_project.py``` file.

## Build a model
In order to train a custom-defined model, you need a Python class to run. Here, are the steps to get it:
1. Generate a new python file in ```src_dnn/models```
2. Add the imports: ```from torch import Tensor, argmax, flatten, nn```
3. Add the following code segment for a ```nn.Module```
```
# Important notes: Add a custom model name, but it must have a _v<idx> at the end!
class ModelName_v0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_shape = (1, 28, 28)
        # Define the model structure
        self.model = nn.Sequential(
            nn.Linear(784, 10)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = flatten(x, start_dim=1)
        prob = self.model(x)
        return prob, argmax(prob, 1)
```
4. Run the ```run_training.py``` file once in order to generate all config files ther should be a text output that new JSON files are generated and the request to run the python script againt (e.g., adapt and restart)
5. Leave the generated config files for now and run the ```run_training.py``` again. Here, an example model (``` mnist_rtg_cl_v0 ```) will be deployed and another JSON file should be generated.
7. Finally, select your model by setting it's name in the ``` ConfigClassifier_MNIST.json ``` config file under ```model_name```.
8. Start training by runing the ```run_training.py``` file again.
