import tensorflow as tf


flags = tf.flags

# ------------setting--------------
# paths
flags.DEFINE_string("data_path", "../data/ptb/", "data directory")
flags.DEFINE_string("train_path", None, "train file for test")
flags.DEFINE_string("dev_path", None, "file for validation")
flags.DEFINE_string("test_path", None, "test file for test")
flags.DEFINE_string("c2v_path", None, "c2v path")
flags.DEFINE_string("dict_path", None, "save dicts")
flags.DEFINE_string("save_path", "./save/", "Model output directory.")

# other settings
flags.DEFINE_string("config", "small", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_boolean("first_run", False, "first run")
flags.DEFINE_boolean("continue_run", False, "continue run")
flags.DEFINE_boolean("shuffle_batch", False, "shffule batch or not")

# ------------config---------------
# parameters to override config
flags.DEFINE_float("init_scale", None, "init scale")
flags.DEFINE_integer("num_layers", None, "num of layers")
flags.DEFINE_float("keep_prob", None, "keep prob")
flags.DEFINE_integer("num_steps", None, "num of steps")
flags.DEFINE_integer("vocab_size", None, "how many words")
flags.DEFINE_float("learning_rate", None, "learning rate")
flags.DEFINE_float("lr_decay", None, "the decay of the learning rate")
flags.DEFINE_integer("max_epoch", None, "every $num epoch reduce learning rate")
flags.DEFINE_integer("max_max_epoch", None, "all epoch rounds")
flags.DEFINE_integer("embedding_size", None, "embedding size word")
flags.DEFINE_integer("batch_size", None, "batch size")
flags.DEFINE_integer("hidden_size", None, "hidden unit size")
flags.DEFINE_string("optimizer_name", None, "type of optimizer")


FLAGS = flags.FLAGS

class Setting:
    def __init__(self):
        self.data_path = FLAGS.data_path
        self.train_path = FLAGS.train_path
        self.dev_path = FLAGS.dev_path
        self.test_path = FLAGS.test_path
        self.save_path = FLAGS.save_path
        self.c2v_path = FLAGS.c2v_path
        self.first_run = FLAGS.first_run
        self.continue_run = FLAGS.continue_run
        self.shuffle_batch = FLAGS.shuffle_batch
        self.config = FLAGS.config
        self.use_fp16 = FLAGS.use_fp16


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def get_config():
    setting = Setting()
    config = SmallConfig()
    if setting.config == "small":
        config = SmallConfig()
    elif setting.config == "medium":
        config = MediumConfig()
    elif setting.config == "large":
        config = LargeConfig()
    elif setting.config == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

    def set(config_item, flag_item):
        if flag_item:
            config_item = flag_item

    set(config.init_scale, FLAGS.init_scale)
    set(config.batch_size, FLAGS.batch_size)
    set(config.hidden_size, FLAGS.hidden_size)
    set(config.learning_rate, FLAGS.learning_rate)
    set(config.lr_decay, FLAGS.lr_decay)
    set(config.max_epoch, FLAGS.max_epoch)
    set(config.max_max_epoch, FLAGS.max_max_epoch)
    set(config.num_layers, FLAGS.num_layers)
    set(config.num_steps, FLAGS.num_steps)
    set(config.keep_prob, FLAGS.keep_prob)
    set(config.vocab_size, FLAGS.vocab_size)

    return config