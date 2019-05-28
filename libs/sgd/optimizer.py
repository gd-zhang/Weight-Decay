from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import gradient_descent
from libs.kfac import estimator as est


class SGDOptimizer(gradient_descent.GradientDescentOptimizer):
    """
    SGD Optimizer
    """

    def __init__(self,
                 learning_rate,
                 var_list=None,
                 momentum=0.,
                 weight_decay=0.,
                 weight_decay_type="l2",
                 weight_list="all",
                 name="SGD"):

        variables = var_list
        if variables is None:
            variables = tf_variables.trainable_variables()
        self.variables = variables

        weight_decay_type = weight_decay_type.lower()
        legal_weight_decay_types = ["wd", "l2", "fisher"]

        if weight_decay_type not in legal_weight_decay_types:
            raise ValueError("Unsupported weight decay type {}. Must be one of {}."
                             .format(weight_decay_type, legal_weight_decay_types))

        self._momentum = momentum
        self._weight_decay = weight_decay
        self._weight_decay_type = weight_decay_type
        self._weight_list = weight_list

        super(SGDOptimizer, self).__init__(learning_rate, name=name)

    def minimize(self, *args, **kwargs):
        kwargs["var_list"] = kwargs.get("var_list") or self.variables
        if set(kwargs["var_list"]) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super(SGDOptimizer, self).minimize(*args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        # args[1] could be our var_list
        if len(args) > 1:
            var_list = args[1]
        else:
            kwargs["var_list"] = kwargs.get("var_list") or self.variables
            var_list = kwargs["var_list"]
        if set(var_list) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super(SGDOptimizer, self).compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        grads_and_vars = list(grads_and_vars)

        if self._weight_decay > 0.0:
            if self._weight_decay_type == "l2" or self._weight_decay_type == "wd":
                grads_and_vars = self._add_weight_decay(grads_and_vars)

        steps_and_vars = self._compute_update_steps(grads_and_vars)
        return super(SGDOptimizer, self).apply_gradients(steps_and_vars,
                                                         *args, **kwargs)

    def _add_weight_decay(self, vecs_and_vars):
        if self._weight_list == "all":
            print("all")
            return [(vec + self._weight_decay * gen_array_ops.stop_gradient(var), var)
                    for vec, var in vecs_and_vars]
        elif self._weight_list == "last":
            print("last")
            grad_list = []
            for vec, var in vecs_and_vars:
                if 'fc' not in var.name:
                    grad_list.append((vec, var))
                else:
                    grad_list.append(
                        (vec + self._weight_decay *
                         gen_array_ops.stop_gradient(var), var))
            return grad_list
        else:
            print("conv")
            grad_list = []
            for vec, var in vecs_and_vars:
                if 'fc' in var.name:
                    grad_list.append((vec, var))
                else:
                    grad_list.append(
                        (vec + self._weight_decay *
                         gen_array_ops.stop_gradient(var), var))
            return grad_list

    def _compute_update_steps(self, grads_and_vars):
        return self._update_velocities(grads_and_vars, self._momentum)

    def _update_velocities(self, vecs_and_vars, decay, vec_coeff=1.0):
        def _update_velocity(vec, var):
            velocity = self._zeros_slot(var, "velocity", self._name)
            with ops.colocate_with(velocity):
                # Compute the new velocity for this variable.
                new_velocity = decay * velocity + vec_coeff * vec

                # Save the updated velocity.
                return (array_ops.identity(velocity.assign(new_velocity)), var)

        # Go through variable and update its associated part of the velocity vector.
        return [_update_velocity(vec, var) for vec, var in vecs_and_vars]
