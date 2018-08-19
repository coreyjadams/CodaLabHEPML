import time

from networks import io_manager

def profile_read(batch_size, n_reads=100):

    config = {
        'FILE' : "/home/cadams/DeepLearnPhysics/CodaLabHEPML/data/train_5-6.h5",
        'BATCH_SIZE' : batch_size,
        'NUM_THREADS': 4,
        'MIN': 0,
        'PREPROCESS': False,
        'MAX': 50,
    }

    # File to read:
    io = io_manager.file_reader(config)
    io.start_reader()


    start = time.time()

    for n in range(n_reads):
        print "Reading " + str(n)
        _ = io.consume_batch_data()

    end = time.time()

    print "Time to read {} batches of size {}: {}".format(n_reads, batch_size, end-start)


if __name__ == '__main__':
    profile_read(4, 10)