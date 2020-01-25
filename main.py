import os
import sys
from RECN import RECN
from Trainer import Trainer


if __name__ == "__main__":
    model = RECN()

    trainer = Trainer(model)

    trainer.load_dataset(os.path.join(sys.path[0], 'data/'))

    trainer.train()

