from denspp.offline.dnn import DefaultSettingsTraining, PyTorchTrainer
from denspp.offline.dnn.models import mnist
from pathlib import Path


if __name__ == "__main__":
    trainer = PyTorchTrainer(
        use_case="MNIST",
        settings=DefaultSettingsTraining,
        default_model=mnist.mnist_mlp_cl_v0.__name__,
        path2config=Path("config"),
    )
    trainer.do_plot_dataset(Path("runs"))
    results = trainer.do_training()
    trainer.do_plot_results(results[0])
