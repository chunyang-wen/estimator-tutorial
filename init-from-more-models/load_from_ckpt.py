import os
import tensorflow as tf

os.makedirs("./models/a", exist_ok=True)
os.makedirs("./models/b", exist_ok=True)

with tf.Session(graph=tf.Graph()) as session:
    tf.Variable(3, name="a")
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    saver.save(session, "./models/a/model-a")


with tf.Session(graph=tf.Graph()) as session:
    tf.Variable(4, name="b")
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    saver.save(session, "./models/b/model-b")


with tf.Session(graph=tf.Graph()) as session:
    a = tf.Variable(1, name="a")
    b = tf.Variable(1, name="b")
    tf.train.init_from_checkpoint("./models/a/model-a", {"a": a})
    tf.train.init_from_checkpoint("./models/b/model-b", {"b": b})
    session.run(tf.global_variables_initializer())
    print(session.run(a))
    print(session.run(b))
