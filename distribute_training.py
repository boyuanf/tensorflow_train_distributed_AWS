from datetime import datetime
import os.path
import re
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 配置神经网络的参数。
BATCH_SIZE = 128
TRAINING_STEPS = 2000
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.

MODEL_SAVE_PATH = "/home/ubuntu/Boyuan/DistributedModelSave"
DATA_PATH = "/home/ubuntu/Boyuan/MNIST_Dataset"



# 和异步模式类似的设置flags。
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string(
    'ps_hosts', 'localhost:2221',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2221,tf-ps1:1111" ')
tf.app.flags.DEFINE_string(
    'worker_hosts', 'localhost:2222,localhost:2223',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:2223" ')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


def forward_propagation(X, layer_hidden_nums, training, dropout_rate=0.01, regularizer_scale=0.01):
    """
    Implements the forward propagation for the model

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)

    Returns:
    Z5 -- the output of the last LINEAR unit
    """
    he_init = tf.contrib.layers.variance_scaling_initializer()
    l1_regularizer = tf.contrib.layers.l1_regularizer(regularizer_scale)

    A_drop = X
    for layer_index, layer_neurons in enumerate(layer_hidden_nums[:-1]):
        Z = tf.layers.dense(inputs=A_drop, units=layer_neurons, kernel_initializer=he_init,
                            kernel_regularizer=l1_regularizer, name="hidden%d" % (layer_index + 1))
        #Z_nor = tf.layers.batch_normalization(Z, training=training, momentum=0.9)
        A = tf.nn.elu(Z)
        A_drop = tf.layers.dropout(A, dropout_rate, training=training, name="hidden%d_drop" % (layer_index + 1))

    # don't do normalization for the output layer
    Z_output = tf.layers.dense(inputs=A_drop, units=layer_hidden_nums[-1], kernel_initializer=he_init, name="output")

    return Z_output


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  #print("labels: ", labels)
  #print("logits: ", logits)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(images, labels):
    """Calculate the total loss on a single tower running the DNN model."""
    # Build inference Graph.
    layer_hidden_nums = [200, 100, 50, 25, 10]
    logits = forward_propagation(images, layer_hidden_nums, True)
    _ = loss(logits, labels)
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    '''
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    '''
    return total_loss, correct

# 和异步模式类似的定义TensorFlow的计算图。唯一的区别在于使用
# tf.train.SyncReplicasOptimizer函数处理同步更新。
def train(images, labels, n_workers, is_chief):

    global_step = tf.train.get_or_create_global_step()
    #print("global_step in train(): ", global_step)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (60000 / BATCH_SIZE)
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) # learning rate decay every NUM_EPOCHS_PER_DECAY epoch
    decay_steps = int(num_batches_per_epoch)  # learning rate decay every epoch

    loss, correct = tower_loss(images, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    # only chief worker will save the graph, so add the task_id is not useless
    loss_name = "loss_%d" % FLAGS.task_id
    tf.summary.scalar(loss_name, loss)

    accuracy_name = "accuracy_%d" % FLAGS.task_id
    tf.summary.scalar(accuracy_name, accuracy)

    # Decay the learning rate exponentially based on the number of steps.
    # decayed_learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)  # use stair learning rate decay or gradually decay

    # 通过tf.train.SyncReplicasOptimizer函数实现同步更新。
    opt = tf.train.SyncReplicasOptimizer(
        tf.train.GradientDescentOptimizer(learning_rate),
        replicas_to_aggregate=n_workers,
        total_num_replicas=n_workers)
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    train_op = opt.minimize(loss, global_step=global_step)
    '''
    if is_chief:
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()
    '''
    return global_step, loss, accuracy, train_op, sync_replicas_hook


def main(argv=None):
    # 和异步模式类似的创建TensorFlow集群。
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    n_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()

    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH)

    device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)

    with tf.device(device_setter):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, ], name='y-input')
        global_step, loss, accuracy, train_op, sync_replicas_hook = train(x, y_, n_workers, is_chief)

        # 把处理同步更新的hook也加进来。
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)

        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        check_point_dir = "{}/run-{}-checkpoint".format(MODEL_SAVE_PATH, now)

        # 训练过程和异步一致。
        # tf.train.MonitoredTrainingSession will automatically continue the training from checkpoint if interrupted.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=check_point_dir,
                                               hooks=hooks,
                                               save_checkpoint_secs=60, # set to None then only save events file
                                               #save_checkpoint_secs=None,  # set to None then only save events file
                                               config=sess_config) as mon_sess:
            print("session started")
            step = 0
            start_time = time.time()

            # The local step is not bound to the global step, the local step will just execute on its own,
            # runs all the time to execute the training model, will stop based on the return of the global step
            # and the training model will return global step based on its status
            while not mon_sess.should_stop():
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, accuracy_value, global_step_value = mon_sess.run(
                    [train_op, loss, accuracy, global_step], feed_dict={x: xs, y_: ys})
                #print("local_step: ", step)
                #print("global_step: ", global_step_value)
                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    format_str = "After %d training steps (%d global steps), " + \
                                 "loss on training batch is %g, " + \
                                 "accuracy is %g. (%.3f sec/batch)"
                    print(format_str % (step, global_step_value, loss_value, accuracy_value, sec_per_batch))
                step += 1

            print("total step: %d, global_step: %d" % (step, global_step_value))


if __name__ == "__main__":
    tf.app.run()