#!/usr/bin/env python
import yaml
import sys

# import all of the possible trainers:
from networks import uresnet_trainer


def main(params):


    trainer = uresnet_trainer.uresnet_trainer(params)

    trainer.initialize()
    trainer.batch_process()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.stdout.write('Requires configuration file.  [train.py config.yml]\n')
        sys.stdout.flush()
        exit()

    config = sys.argv[-1]

    with open(config, 'r') as f:
        params = yaml.load(f)

    main(params)