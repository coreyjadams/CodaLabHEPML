base_config = '''
# network parameters
NAME: {name}

#
# Data is 192x192x192, 192 == 2**6 * 3
# Downsampling 4 times gives a spatial resolution of 12 x 12 x 12

NETWORK:
  N_INITIAL_FILTERS:  8
  NETWORK_DEPTH:  4
  RESIDUAL: {block}
  CONCATENATE: {concat}
  MERGE_BEFORE_BLOCK: {merge}
  BLOCKS_PER_LAYER: 6
  BLOCKS_DEEPEST_LAYER: 8
  BALANCE_LOSS: True
  BATCH_NORM: True
  NUM_LABELS: 4
  REGULARIZE: 0.001

# training parameters:
N_MINIBATCH: 4
MINIBATCH_SIZE: 2
SAVE_ITERATION: 500
REPORT_ITERATION: 10
SUMMARY_ITERATION: 1
BASE_LEARNING_RATE:  0.001
TRAINING: True
ITERATIONS: 25000

# IO Parameters:
IO:
  TRAIN:
    FILE: './larcv_io_config/train_io.cfg'
    FILLER: 'TrainIO'
    VERBOSITY: 3
    KEYWORD_DATA:  'main_data'
    KEYWORD_LABEL: 'main_label'

  TEST:
    PROFILE_IO: false
    FILE: './larcv_io_config/test_io.cfg'
    FILLER: 'TestIO'
    VERBOSITY: 3
    KEYWORD_DATA:  'test_data'
    KEYWORD_LABEL: 'test_label'

# General parameters:
LOGDIR: 'log/{name}/'
RESTORE: False



'''

block_options = ['True', 'False']
concat_options = ['True', 'False']
merge_options = ['True', 'False']

for block_choice in block_options:
    for concat_choice in concat_options:
        for merge_choice in merge_options:
            name = "unet_3d_concat_{}_residual_{}_merge_pre_block_{}".format(concat_choice, block_choice, merge_choice)

            config = base_config.format(
                    name = name,
                    block = block_choice,
                    concat = concat_choice,
                    merge = merge_choice
                )
            with open(name + ".yml", 'w') as _cfg:
                _cfg.write(config)