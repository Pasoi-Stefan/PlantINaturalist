import pytest
import PlantINaturalist2021DataModule
import numpy

NUM_CLASSES = 250
NUM_TRAINING = 40
NUM_FINETUNE = 10
NUM_VALIDATION = 10

def test_trainDataloader():
    for batch_sz in [1, 2, 4, 20, 250]:
        dataloader = PlantINaturalist2021DataModule.PlantINaturalist2021DataModule(PlantINaturalist2021DataModule.TRANSFORM, context="train", batch_size=batch_sz)
        dataloader.setup("fit")
        assert(len(dataloader.train_dataloader()) == NUM_CLASSES * NUM_TRAINING / batch_sz)
        assert(len(dataloader.val_dataloader()) == NUM_CLASSES*NUM_VALIDATION / batch_sz)


def test_finetuneDataloader():
    for batch_sz in [1, 2, 4, 20, 250]:
        dataloader = PlantINaturalist2021DataModule.PlantINaturalist2021DataModule(PlantINaturalist2021DataModule.TRANSFORM, context="finetune", batch_size=batch_sz)
        dataloader.setup("fit")
        assert(len(dataloader.train_dataloader()) == NUM_CLASSES * NUM_FINETUNE / batch_sz)
        assert(len(dataloader.val_dataloader()) == NUM_CLASSES*NUM_VALIDATION / batch_sz)

def test_retrainDataloader():
    for batch_sz in [1, 2, 4, 20, 250]:
        dataloader = PlantINaturalist2021DataModule.PlantINaturalist2021DataModule(PlantINaturalist2021DataModule.TRANSFORM, context="retrain", batch_size=batch_sz)
        dataloader.setup("fit")
        assert(len(dataloader.train_dataloader()) == NUM_CLASSES * (NUM_TRAINING + NUM_FINETUNE) / batch_sz)
        assert(len(dataloader.val_dataloader()) == NUM_CLASSES*NUM_VALIDATION / batch_sz)


def test_Dataloader():
    dataloader = PlantINaturalist2021DataModule.PlantINaturalist2021DataModule(PlantINaturalist2021DataModule.TRANSFORM, context="train", batch_size=1)
    dataloader.setup("fit")
    presence = numpy.zeros(250)
    iterator = iter(dataloader.train_dataloader())
    for _ in range(NUM_CLASSES * NUM_TRAINING):
        image_batch = next(iterator, None)
        if image_batch == None:
            break
        assert numpy.array(image_batch[0]).shape == (1, 3, 256, 256)
        assert 0 <= image_batch[1] and image_batch[1] < 250
        presence[image_batch[1]] += 1
    
    for clas in range(0, 250):
        assert presence[clas] == NUM_TRAINING
