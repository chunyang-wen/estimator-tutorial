import logging
import os
import random
import subprocess

import tensorflow as tf

from model import MyEstimator


logging.getLogger().setLevel(logging.INFO)

model_dir = "/tmp/temp_model_dir/"
subprocess.check_call("rm -rf %s" % model_dir, shell=True)

estimator = MyEstimator(model_dir)

batch_size = 1

def train_input_fn():
    def generator():
        for _ in range(10):
            datum = random.random()
            yield "\t".join(map(str, (datum, datum * 0.8 + 1)))

    def parse(line):
        fields = tf.decode_csv(line, [[0.0], [0.0]], field_delim="\t")
        return {"x": fields[0]}, fields[1]

    dataset = tf.data.Dataset.from_generator(
        generator, tf.string, tf.TensorShape([])
    )
    dataset = dataset.map(parse)
    return dataset.batch(batch_size)


def serving_input_fn():
    feature_tensors = {
        "x": tf.placeholder(tf.float32, shape=(None, 1), name="input_x")
    }
    receiver_tensor = tf.placeholder(
        tf.float32, shape=(None, 1), name="output_tensor"
    )
    return tf.estimator.export.ServingInputReceiver(
        feature_tensors, receiver_tensor
    )


def predict_input_fn():
    def generator():
        for _ in range(10):
            datum = random.random()
            yield "\t".join(map(str, (datum,)))

    def parse(line):
        fields = tf.decode_csv(line, [[0.0]], field_delim="\t")
        return {"x": fields[0]}

    dataset = tf.data.Dataset.from_generator(
        generator, tf.string, tf.TensorShape([])
    )
    dataset = dataset.map(parse)
    return dataset.batch(batch_size)


estimator.train(train_input_fn)
estimator.evaluate(train_input_fn)
base = os.path.join(model_dir, "test")
result_dir = estimator.export_savedmodel(base, serving_input_fn)
print("Result dir: ", result_dir)

for data in estimator.predict(predict_input_fn):
    print(data)
