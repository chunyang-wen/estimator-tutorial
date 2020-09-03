import argparse
import json
import logging
import os
import random
import sys
import subprocess

import tensorflow as tf

from model import MyEstimator


logging.getLogger().setLevel(logging.INFO)

model_dir = "/tmp/temp_model_dir/"
subprocess.check_call("rm -rf %s" % model_dir, shell=True)


batch_size = 1
train_number = 1000
test_number = 100

def input_fn(data_size):
    def actual_input_fn():
        def generator():
            for _ in range(data_size):
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
    return actual_input_fn


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

train_spec = tf.estimator.TrainSpec(
    input_fn(train_number), max_steps=500, hooks=None
)
eval_spec = tf.estimator.EvalSpec(
    input_fn(test_number), steps=50, name=None, hooks=None, exporters=None,
    start_delay_secs=0, throttle_secs=0
)

def get_cluster(args):
    """get_cluster"""
    cluster = {
        "cluster": {
            "ps": args.ps_hosts.split(";"),
            "worker": args.worker_hosts.split(";"),
            "chief": args.chief_hosts.split(";"),
        },
        "task": {
            "type": args.worker_type,
            "index": args.worker_index,
        }
    }
    os.environ["TF_CONFIG"] = json.dumps(cluster)

parser = argparse.ArgumentParser()
parser.add_argument("--ps-hosts")
parser.add_argument("--worker-hosts")
parser.add_argument("--chief-hosts")
parser.add_argument("--evaluator")
parser.add_argument("--worker-type", type=str)
parser.add_argument("--worker-index", type=int)

print("Argv: ", sys.argv)
args, _ = parser.parse_known_args()

get_cluster(args)

estimator = MyEstimator(model_dir)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

