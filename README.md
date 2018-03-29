# tensorflow_train_distributed_AWS

On single server cmd:

CUDA_VISIBLE_DEVICES='' python distribute_training.py --job_name=ps --task_id=0 --ps_hosts=localhost:2221 --worker_hosts=localhost:2222,localhost:2223

python distribute_training.py --job_name=worker --task_id=0 --ps_hosts=localhost:2221 --worker_hosts=localhost:2222,localhost:2223

python distribute_training.py --job_name=worker --task_id=1 --ps_hosts=localhost:2221 --worker_hosts=localhost:2222,localhost:2223


Multiple server:

CUDA_VISIBLE_DEVICES='' python distribute_training.py --job_name=ps --task_id=0 --ps_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2221 --worker_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2222,ec2-52-77-218-2.ap-southeast-1.compute.amazonaws.com:2223

python distribute_training.py --job_name=worker --task_id=0 --ps_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2221 --worker_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2222,ec2-52-77-218-2.ap-southeast-1.compute.amazonaws.com:2223

python distribute_training.py --job_name=worker --task_id=1 --ps_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2221 --worker_hosts=ec2-54-169-168-26.ap-southeast-1.compute.amazonaws.com:2222,ec2-52-77-218-2.ap-southeast-1.compute.amazonaws.com:2223


## 1. Single thread reading file:

The training speed of two GPU server (red, each have one GPU) compare with one GPU server (blue, one GPU), the data is read by single thread using mnist.train.next_batch(BATCH_SIZE).



Output of one of the two GPU server:
session started
After 100 training steps (95 global steps), loss on training batch is 0.811832, accuracy .789062. (0.014 sec/batch)
After 200 training steps (195 global steps), loss on training batch is 0.716209, accuracy0.796875. (0.012 sec/batch)
After 300 training steps (295 global steps), loss on training batch is 0.475263, accuracy0.851562. (0.011 sec/batch)
After 400 training steps (395 global steps), loss on training batch is 0.490671, accuracy0.859375. (0.011 sec/batch)
After 500 training steps (495 global steps), loss on training batch is 0.366699, accuracy0.890625. (0.011 sec/batch)
After 600 training steps (595 global steps), loss on training batch is 0.356246, accuracy0.867188. (0.010 sec/batch)
After 700 training steps (695 global steps), loss on training batch is 0.380742, accuracy0.875. (0.010 sec/batch)
After 800 training steps (795 global steps), loss on training batch is 0.265325, accuracy0.890625. (0.010 sec/batch)
After 900 training steps (895 global steps), loss on training batch is 0.322615, accuracy0.898438. (0.010 sec/batch)
After 1000 training steps (995 global steps), loss on training batch is 0.257241, accurac 0.921875. (0.010 sec/batch)
After 1100 training steps (1095 global steps), loss on training batch is 0.400545, accuras 0.882812. (0.010 sec/batch)
After 1200 training steps (1195 global steps), loss on training batch is 0.339254, accuras 0.882812. (0.010 sec/batch)
After 1300 training steps (1295 global steps), loss on training batch is 0.311921, accuras 0.882812. (0.010 sec/batch)
After 1400 training steps (1395 global steps), loss on training batch is 0.255179, accuras 0.929688. (0.010 sec/batch)
After 1500 training steps (1495 global steps), loss on training batch is 0.308558, accuras 0.9375. (0.010 sec/batch)
After 1600 training steps (1595 global steps), loss on training batch is 0.366522, accuras 0.867188. (0.010 sec/batch)
After 1700 training steps (1695 global steps), loss on training batch is 0.232567, accuras 0.929688. (0.010 sec/batch)
After 1800 training steps (1795 global steps), loss on training batch is 0.190541, accuras 0.945312. (0.010 sec/batch)
After 1900 training steps (1895 global steps), loss on training batch is 0.235993, accuras 0.9375. (0.010 sec/batch)
After 2000 training steps (1995 global steps), loss on training batch is 0.344351, accuras 0.921875. (0.010 sec/batch)
total step: 2005, global_step: 1999

Output of one GPU server:
session started
After 100 training steps (95 global steps), loss on training batch is 0.929126, accuracy is 0.726562. (0.021 sec/batch)
After 200 training steps (195 global steps), loss on training batch is 0.578736, accuracy is 0.84375. (0.017 sec/batch)
After 300 training steps (295 global steps), loss on training batch is 0.426105, accuracy is 0.875. (0.015 sec/batch)
After 400 training steps (395 global steps), loss on training batch is 0.438998, accuracy is 0.882812. (0.015 sec/batch)
After 500 training steps (495 global steps), loss on training batch is 0.373946, accuracy is 0.898438. (0.015 sec/batch)
After 600 training steps (595 global steps), loss on training batch is 0.407651, accuracy is 0.851562. (0.014 sec/batch)
After 700 training steps (695 global steps), loss on training batch is 0.301445, accuracy is 0.914062. (0.014 sec/batch)
After 800 training steps (795 global steps), loss on training batch is 0.328147, accuracy is 0.875. (0.014 sec/batch)
After 900 training steps (895 global steps), loss on training batch is 0.344807, accuracy is 0.890625. (0.014 sec/batch)
After 1000 training steps (995 global steps), loss on training batch is 0.303609, accuracy is 0.921875. (0.014 sec/batch)
After 1100 training steps (1095 global steps), loss on training batch is 0.328098, accuracy is 0.90625. (0.014 sec/batch)
After 1200 training steps (1195 global steps), loss on training batch is 0.248056, accuracy is 0.945312. (0.014 sec/batch)
After 1300 training steps (1295 global steps), loss on training batch is 0.368613, accuracy is 0.882812. (0.014 sec/batch)
After 1400 training steps (1395 global steps), loss on training batch is 0.257727, accuracy is 0.914062. (0.014 sec/batch)
After 1500 training steps (1495 global steps), loss on training batch is 0.20555, accuracy is 0.9375. (0.014 sec/batch)
After 1600 training steps (1595 global steps), loss on training batch is 0.25864, accuracy is 0.921875. (0.014 sec/batch)
After 1700 training steps (1695 global steps), loss on training batch is 0.230667, accuracy is 0.929688. (0.014 sec/batch)
After 1800 training steps (1795 global steps), loss on training batch is 0.225652, accuracy is 0.960938. (0.014 sec/batch)
After 1900 training steps (1895 global steps), loss on training batch is 0.443253, accuracy is 0.84375. (0.014 sec/batch)
After 2000 training steps (1995 global steps), loss on training batch is 0.233018, accuracy is 0.9375. (0.014 sec/batch)
total step: 2005, global_step: 1999

## 2. Use multi-thread reading data

Instead of read dataset by sequence, using queue and thread to read data, helps training faster (rose red line). (the queue method not work on one PC, may be because the CPU power has already used by computing, not enough for data reading thread)

The output of two servers:

session started
After 100 training steps (158 global steps), loss on training batch is 0.697824, accuracy is 0.757812. (0.018 sec/batch)
After 200 training steps (385 global steps), loss on training batch is 0.553205, accuracy is 0.835938. (0.012 sec/batch)
After 300 training steps (610 global steps), loss on training batch is 0.413707, accuracy is 0.84375. (0.011 sec/batch)
After 400 training steps (833 global steps), loss on training batch is 0.315509, accuracy is 0.90625. (0.011 sec/batch)
After 500 training steps (1056 global steps), loss on training batch is 0.344955, accuracy is 0.914062. (0.010 sec/batch)
After 600 training steps (1282 global steps), loss on training batch is 0.378431, accuracy is 0.890625. (0.010 sec/batch)
After 700 training steps (1505 global steps), loss on training batch is 0.288738, accuracy is 0.898438. (0.010 sec/batch)
After 800 training steps (1721 global steps), loss on training batch is 0.197968, accuracy is 0.921875. (0.010 sec/batch)
After 900 training steps (1948 global steps), loss on training batch is 0.16924, accuracy is 0.953125. (0.010 sec/batch)
total step: 923, global_step: 1999


session started
After 100 training steps (232 global steps), loss on training batch is 0.627283, accuracy is 0.828125. (0.010 sec/batch)
After 200 training steps (410 global steps), loss on training batch is 0.558948, accuracy is 0.804688. (0.009 sec/batch)
After 300 training steps (592 global steps), loss on training batch is 0.420959, accuracy is 0.859375. (0.009 sec/batch)
After 400 training steps (774 global steps), loss on training batch is 0.379158, accuracy is 0.867188. (0.009 sec/batch)
After 500 training steps (954 global steps), loss on training batch is 0.424409, accuracy is 0.867188. (0.009 sec/batch)
After 600 training steps (1135 global steps), loss on training batch is 0.321105, accuracy is 0.898438. (0.009 sec/batch)
After 700 training steps (1313 global steps), loss on training batch is 0.237776, accuracy is 0.90625. (0.009 sec/batch)
After 800 training steps (1495 global steps), loss on training batch is 0.191302, accuracy is 0.953125. (0.009 sec/batch)
After 900 training steps (1683 global steps), loss on training batch is 0.263927, accuracy is 0.914062. (0.009 sec/batch)
After 1000 training steps (1862 global steps), loss on training batch is 0.203689, accuracy is 0.914062. (0.009 sec/batch)
total step: 1078, global_step: 1999


Even we set reading using only machine0's CPU, it seems machine1 will also read data on its local, although the variable is located on machine1, so the data has to be put in both machine, in order to read data.

