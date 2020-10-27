import argparse
import logging
import os
import random
import gc
import subprocess
import time
import sys
import threading
from multiprocessing import Process
from tensorflow.python.training import server_lib
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook


import tensorflow as tf
from tensorflow.python import keras as K

import psutil
import objgraph


class MyEstimator(tf.estimator.Estimator):
    """MyEstimator"""

    def __init__(self, model_dir, config=None, params=None):
        super(MyEstimator, self).__init__(
            self.model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
        )

    def model_fn(self, features, labels, mode, config):
        # 具体的含义见
        # https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#args
        optimizer = tf.train.AdamOptimizer()
        x = features["x"]
        w = tf.Variable(0.1, name="x")
        b = tf.Variable(0.1, name="b")
        gap = tf.Variable([1]*1000000, name="gap")
        prediction = w * x + b
        print("Mode = ", mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)

        loss = tf.losses.mean_squared_error(labels, prediction)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step()
        )
        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {
                "mse": tf.metrics.mean_squared_error(labels, prediction)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=prediction,
                eval_metric_ops=metrics,
                loss=loss,
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode, predictions=prediction, loss=loss, train_op=train_op,
            )

        raise ValueError("Not a valid mode: {}".format(mode))



def main():
    pid = os.getpid()
    print("pid: ", pid)

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["train", "eval", "export"])
    args, _ = parser.parse_known_args()

    logging.getLogger().setLevel(logging.INFO)
    model_dir = "/tmp/temp_model_dir/"

    if args.action == "train":
        print("clean export dir first and then export: ", model_dir)
        subprocess.check_call("rm -rf %s" % model_dir, shell=True)
        estimator = MyEstimator(model_dir, config=get_config_proto(model_dir))
        estimator.train(train_input_fn)
    elif args.action == "eval":
        base = os.path.join(model_dir, "test")
        subprocess.check_call("rm -rf %s" % base, shell=True)
        for _ in range(100):
            memory_usage(pid, prefix="In parent before")
            estimator = MyEstimator(model_dir, config=get_config_proto(model_dir))
            estimator.evaluate(train_input_fn, hooks=[MyHook()])
            memory_usage(pid, prefix="In parent after")
            memory_usage(pid, prefix="In parent before export")
            target(base, get_config_proto(model_dir), pid)
            memory_usage(pid, prefix="In parent after export")
    elif args.action == "export":
        session = tf.Session()  # 故意创建 Session，模拟真实过程
        base = os.path.join(model_dir, "test")
        subprocess.check_call("rm -rf %s" % base, shell=True)
        use_thread = False  # 是否使用线程
        for _ in range(300):
            memory_usage(pid, prefix="In parent before")
            if use_thread:
                t = threading.Thread(
                    target=target,
                    args=(base, get_config_proto(model_dir), pid),
                )
                t.start()
                t.join()
            else:
                target(base, get_config_proto(model_dir), pid)
            memory_usage(pid, prefix="In parent after")
    else:
        raise ValueError("Wrong action: {}".format(args.action))


def get_config_proto(model_dir):
    NUM_CORES=1
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=NUM_CORES,
        intra_op_parallelism_threads=NUM_CORES,
    )
    config = tf.estimator.RunConfig(model_dir=model_dir, session_config=session_config)
    return config


MB = 1024 * 1024.0


def memory_usage(p, prefix=None):
    p = psutil.Process(p)
    mem_usage = p.memory_info().rss / MB

    # this can be > 100% in case a process running
    # multi-thread in different cpu.
    cpu_usage = p.cpu_percent()

    children = p.children(recursive=True)
    child_info = ""

    for child in children:
        mem_child = child.memory_info().rss / MB
        cpu_child = child.cpu_percent()
        child_info += "(mem:%d, cpu:%.1f%%)" % (mem_child, cpu_child)
    print("=" * 100)
    result = (
        "PID(%d):[mem:%dMB, cpu:%.1f%%], with_child(%d):[%s]" % (
            p.pid,
            mem_usage,
            cpu_usage,
            len(children),
            child_info,
        )
    )
    if prefix:
        result = prefix + "::" + result
    print(result)
    print("=" * 100)
    sys.stdout.flush()


def train_input_fn():
    batch_size = 1
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
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    return iterator


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


def target(export_dir, config, pid):
    print("===== before =====")
    # memory_usage(pid)
    print("=====common types=====")
    objgraph.show_most_common_types()
    print("=====common types=====")
    print("=====growth=====")
    objgraph.show_growth()
    print("=====growth=====")

    print("===== before =====")
    estimator = MyEstimator(model_dir=config.model_dir, config=config)
    result_dir = estimator.export_savedmodel(export_dir, serving_input_fn)
    print("Result dir: ", result_dir)
    time.sleep(1)
    print("Show stats:")
    clean(estimator)
    print("===== after =====")
    # memory_usage(pid)
    print("=====common types=====")
    objgraph.show_most_common_types()
    print("=====common types=====")
    print("=====growth=====")
    objgraph.show_growth()
    print("=====growth=====")
    print("===== after =====")


if __name__ == "__main__":
    main()
