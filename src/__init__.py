import logging
import transformers

# turn off unnecessary logging
transformers.logging.set_verbosity_error()

from .utils import set_seed, Range, TensorBoardRunner, check_args, init_weights, TqdmToLogger, MetricManager
from .loaders import load_dataset, load_model



# for logger initialization
def set_logger(path, args):
    # initialize logger
    logger = logging.getLogger(__name__)
    logging_format = logging.Formatter(
        fmt='[%(levelname)s] (%(asctime)s) %(message)s',
        datefmt='%Y/%m/%d %I:%M:%S %p'
    )
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(path)
    
    stream_handler.setFormatter(logging_format)
    file_handler.setFormatter(logging_format)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level=logging.INFO)
    
    # print welcome message
    logger.info('[WELCOME] Initialize...')
    welcome_message =  """
       _______ _______ ______  |       _____ \   /
       |______ |______ |     \ |      |_____  \ /
       |       |_______|_____/ |_____ |_____  / \ 
       |                                     /   \ 

                    Christian Interno´
       """
    logger.info(welcome_message)
