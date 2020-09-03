import argparse
import json
import logging
import os
import random
import sys
import subprocess

import six
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training import server_lib
from tensorflow.python.training import training
from tensorflow.python.framework import random_seed
from tensorflow.python.eager import context
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import estimator
from tensorflow.python.framework import ops

from model import MyEstimator


logging.getLogger().setLevel(logging.INFO)

model_dir = "/tmp/temp_model_dir/"


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
        return dataset.batch(batch_size).make_one_shot_iterator().get_next()
    return actual_input_fn


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


def run_std_server(config):
    if config.session_config is None:
        session_config = config_pb2.ConfigProto(log_device_placement=False)
    else:
        session_config = config_pb2.ConfigProto(
            log_device_placement=False,
            gpu_options=config.session_config.gpu_options,
        )

        server = server_lib.Server(
            config.cluster_spec,
            job_name=config.task_type,
            task_index=config.task_id,
            config=session_config,
            start=False,
            protocol=config.protocol,
        )
        server.start()
        return server


def hook_predict(args, config):

    # Override estimator predict
    def predict(
        self,
        input_fn,
        predict_keys=None,
        hooks=None,
        checkpoint_dir=None,
        yield_single_examples=True,
    ):
        """Arguments are same with Estimator.predict"""
        with context.graph_mode():
            hooks = estimator._check_hooks_type(hooks)
            # Check that model has been trained.
            if not checkpoint_dir:
                raise ValueError("No checkpoint_dir")
            with ops.Graph().as_default() as g, g.device(self._device_fn):
                random_seed.set_random_seed(self._config.tf_random_seed)
                self._create_and_assert_global_step(g)
                features, input_hooks = self._get_features_from_input_fn(
                    input_fn, model_fn_lib.ModeKeys.PREDICT
                )
                estimator_spec = self._call_model_fn(
                    features,
                    None,
                    model_fn_lib.ModeKeys.PREDICT,
                    self.config,
                )

                predictions = self._extract_keys(
                    estimator_spec.predictions, predict_keys
                )
                all_hooks = list(input_hooks)
                all_hooks.extend(hooks)
                all_hooks.extend(
                    list(estimator_spec.prediction_hooks or [])
                )
                with training.MonitoredTrainingSession(
                    is_chief=args.worker_type=="chief",
                    master=config.master,
                    checkpoint_dir=checkpoint_dir,
                    config=config.session_config,
                ) as mon_sess:

                    while not mon_sess.should_stop():
                        preds_evaluated = mon_sess.run(predictions)
                        if not yield_single_examples:
                            yield preds_evaluated
                        elif not isinstance(predictions, dict):
                            for pred in preds_evaluated:
                                yield pred
                        else:
                            for i in range(
                                self._extract_batch_length(preds_evaluated)
                            ):
                                yield {
                                    key: value[i]
                                    for key, value in six.iteritems(
                                        preds_evaluated
                                    )
                                }
    estimator.Estimator.predict = predict


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

user_estimator = MyEstimator(model_dir)

server = run_std_server(user_estimator.config)

if args.worker_type == "ps":
    server.join()
else:
    hook_predict(args, user_estimator.config)
    kwargs = {
        "checkpoint_dir":  model_dir,
    }
    for data in user_estimator.predict(input_fn(10), **kwargs):
        print(data)
