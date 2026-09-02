from denspp.offline.dnn import DefaultSettingsTraining, PyTorchTrainer
from src_dnn.models.mnist_live import mnist_rtg_v0


if __name__ == "__main__":
    trainer = PyTorchTrainer(
        use_case="MNIST",
        settings=DefaultSettingsTraining,
        default_model=mnist_rtg_v0.__name__
    )
    trainer.do_plot_dataset()
    results = trainer.do_training()
    trainer.do_plot_results(results[0])
