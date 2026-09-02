# Software for RTG Specialized Course - Deep Learning

## 1) Introduction
This repo helps you to guide through the exercise to train deep neural networks in Python.
Here, we will use the *PyTorch* framework inside the *denspp.offline* framework. You find further information [here](https://github.com/es-ude/denspp.offline).
Software for RTG Specialized Course - Deep Learning

## 2) Installation Guide
### Tools
To use this software, we recommend to install the following tools:
- uv package manager ([Link](https://docs.astral.sh/uv/) for installation)
- VScode ([Link](https://code.visualstudio.com) for Downloading and Installation)
- Git ([Link](https://git-scm.com/) for Downloading and Installation)

### 3) Initialisation
To download the code open a terminal/cmd and navigate to the target directory for example: <br>
```cd C:\Users\Student\Documents\GitHub``` <br>
Then clone the repository (e.g., download a copy of the code): <br>
```git clone https://github.com/AErbsloeh/rtg_deeplearning``` <br>
This will automatically create a subfolder in your directory named rtg_deeplearning. Go to your code editor of choice (e.g. VScode) and navigate to this folder. <br>
To use the code, you have to initialise the project. Please run the following steps:
- Please enter ```uv sync``` into the terminal in order to install all packages.
- After it, please run the ```init_project.py``` file.

## 4) Building a model
In order to train a custom-defined model, you need a Python class to run. Here, are the steps to get it:
1. Generate a new Python file in ```src_dnn/models```
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
5. Leave the generated config files for now and run the ```run_training.py``` again. Here, an example model will be deployed and another JSON file should be generated.
7. Finally, select your model by setting it's name in the ``` ConfigClassifier_MNIST.json ``` config file under ```model_name```.
8. Start training by runing the ```run_training.py``` file again.


## 5) Exercises
In order to build and train custom neural networks, we will use the MNIST dataset (link)[https://en.wikipedia.org/wiki/MNIST_database]. 
It includes 70,000 figures / samples with handwritten numbers (shape [28 x 28 pixels] in grayscale). 
Also, each sample has a label for the corresponding number (0-9).

During this exercise, we will train a simple model for this benchmark and enhance it until we get a lightweight and stable model.
All layer functionalities are in the ```torch.nn``` library.

Please try to extract the performance (accuracy, overfitting), extract the number of parameters for each model and explain the changes on the model's performance. 

### 5.1) Exercises using MLPs:
1. Build and train a simple model with one ```Linear``` layer (768 -> 10). Are any further layers/functions necessary?
2. Please add an activation function like ```PReLU``` or ```ReLU```.
3. Please add a second computing block of ```Linear``` and ```PReLU/ReLU``` (768 -> 256 -> 10).
4. Please add a third computing block of ```Linear``` and ```PReLU/ReLU``` (768 -> 256 -> 96 -> 10).
5. Please add a ```Dropout``` after the last activation function
6. Please add a ```BatchNorm1d``` after the first ```Linear``` layer.
7. Please add a fourth computing block of ```Linear```, ```BatchNorm1d``` and ```PReLU/ReLU``` (768 -> 512 -> 256 -> 96 -> 10).

### 5.2) Exercises using CNNs:
In this case you need a CNN for feature extraction and a MLP for classification:
Hint: We can talk a ```Conv2d```, the following configuration is given in breakets: (num. filters, kernel size, stride, padding).
1. Build and train a simple model with one ```Conv2d``` (16, 4, 2, 1) layer, ```ReLU```, and ```LazyLinear``` (10) for classification. Are any further layers/functions necessary?
2. Please add a pooling like ```MaxPool2d``` layer with (kernel=3, stride=2).
3. Please add a second computing block with ```Conv2d```(64, 3, 1, 0) and ```MaxPool2d``` (2, 1).
4. Please add a ```BatchNorm2d``` after each ```Conv2d``` layer.
5. Please add a ```Dropout2d``` (p=15%) before the last Conv2d layer.
6. Please add in the MLP sequential a computing layer with ```Linear``` (Lazy -> 128 -> 10), ```BatchNorm1d``` and ```ReLU```.
