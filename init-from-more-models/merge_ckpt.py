import tensorflow as tf
from tensorflow.python.ops import gen_io_ops

src = tf.constant(["./models/a/model-a", "./models/b/model-b"])
target = tf.constant("./models/merged_model")

op = gen_io_ops.merge_v2_checkpoints(src, target)

tf.Session().run(op)

with tf.Session(graph=tf.Graph()) as session:

    a = tf.Variable(1, name="a")
    b = tf.Variable(1, name="b")
    saver = tf.train.Saver()
    saver.restore(session, "./models/merged_model")
    print(session.run(a))
    print(session.run(b))
