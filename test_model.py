import PlantINaturalist2021FinetuneMobileNetv2
import torch
import numpy

CONFIG = {
    "model_name": PlantINaturalist2021FinetuneMobileNetv2.__name__,
    "num_classes": 250,
    "learning_rate": 0.01,
    "lr_decay_epoch_step_size": 5,
    "lr_decay_rate": 0.9,
    "num_trainable_layers": 2,
}
def test_model():
    model = PlantINaturalist2021FinetuneMobileNetv2.PlantINaturalist2021FinetuneMobileNetv2(CONFIG)
    assert model.num_classes == 250
    assert model.learning_rate == 0.01
    assert model.lr_decay_epoch_step_size == 5
    assert model.lr_decay_rate == 0.9
    assert model.num_trainable_layers == 2

    data = torch.Tensor(numpy.random.rand(5, 3, 256, 256))
    preds = model.model(data)
    assert preds.shape == (5, 250)
