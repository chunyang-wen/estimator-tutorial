import tensorflow as tf

"""
Estimator interface
tf.estimator.Estimator(
    model_fn,
    model_dir=None,
    config=None,
    params=None,
    warm_start_from=None,
)
"""


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
