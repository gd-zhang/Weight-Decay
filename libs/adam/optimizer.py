from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import gradient_descent


class ADAMOptimizer(gradient_descent.GradientDescentOptimizer):
    """
    ADAM Optimizer
    """

    def __init__(self,
                 learning_rate,
                 beta1=0.9,
                 beta2=0.999,
                 var_list=None,
                 epsilon=1e-8,
                 weight_decay=0.,
                 weight_decay_type="l2",
                 weight_list="all",
                 name="ADAM"):

        variables = var_list
        if variables is None:
            variables = tf_variables.trainable_variables()
        self.variables = variables

        weight_decay_type = weight_decay_type.lower()
        legal_weight_decay_types = ["wd", "l2"]

        if weight_decay_type not in legal_weight_decay_types:
            raise ValueError("Unsupported weight decay type {}. Must be one of {}."
                             .format(weight_decay_type, legal_weight_decay_types))

        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._weight_decay_type = weight_decay_type
        self._weight_list = weight_list
        self._init()

        super(ADAMOptimizer, self).__init__(learning_rate, name=name)

    def minimize(self, *args, **kwargs):
        kwargs["var_list"] = kwargs.get("var_list") or self.variables
        if set(kwargs["var_list"]) != set(self.variables):
            raise ValueError("var_list doesn't match with set of Fisher-estimating "
                             "variables.")
        return super(ADAMOptimizer, self).minimize(*args, **kwargs)

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
        return super(ADAMOptimizer, self).compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        grads_and_vars = list(grads_and_vars)

        if self._weight_decay > 0.0:
            if self._weight_decay_type == "l2":
                grads_and_vars = self._add_weight_decay(grads_and_vars)

        velocities_and_vars = self._update_velocities(grads_and_vars, self._beta1)
        covariances_and_vars = self._update_covariances(grads_and_vars, self._beta2)

        beta1_update_op = self._beta1_power.assign(self._beta1_power * self._beta1)
        beta2_update_op = self._beta2_power.assign(self._beta2_power * self._beta2)
        with ops.control_dependencies([beta1_update_op, beta2_update_op]):
            steps_and_vars = self._compute_update_steps(velocities_and_vars, covariances_and_vars)
            if self._weight_decay_type == "wd" and self._weight_decay > 0.0:
                steps_and_vars = self._add_weight_decay(steps_and_vars)
            update_ops = super(ADAMOptimizer, self).apply_gradients(steps_and_vars,
                                                                    *args, **kwargs)
        return update_ops

    def _compute_update_steps(self, velocities_and_vars, covariances_and_vars):
        steps_and_vars = []
        for (velo, var1), (covar, var2) in zip(velocities_and_vars, covariances_and_vars):
            if var1 is not var2:
                raise ValueError("The variables referenced by the two arguments "
                                 "must match.")
            velo = velo / (1 - self._beta1_power)
            covar = covar / (1 - self._beta2_power)
            step = velo / (math_ops.sqrt(covar) + self._epsilon)
            steps_and_vars.append((step, var1))

        return steps_and_vars

    def _init(self):
        first_var = min(self.variables, key=lambda x: x.name)
        with ops.colocate_with(first_var):
            self._beta1_power = variable_scope.variable(1.0,
                                                        name="beta1_power",
                                                        trainable=False)
            self._beta2_power = variable_scope.variable(1.0,
                                                        name="beta2_power",
                                                        trainable=False)

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

    def _update_velocities(self, vecs_and_vars, decay):
        def _update_velocity(vec, var):
            velocity = self._zeros_slot(var, "velocity", self._name)
            with ops.colocate_with(velocity):
                # Compute the new velocity for this variable.
                new_velocity = decay * velocity + (1 - decay) * vec

                # Save the updated velocity.
                return (array_ops.identity(velocity.assign(new_velocity)), var)

        # Go through variable and update its associated part of the velocity vector.
        return [_update_velocity(vec, var) for vec, var in vecs_and_vars]

    def _update_covariances(self, vecs_and_vars, decay):
        def _update_covariance(vec, var):
            covariance = self._zeros_slot(var, "covariance", self._name)
            with ops.colocate_with(covariance):
                # Compute the new velocity for this variable.
                new_covariance = decay * covariance + (1 - decay) * vec ** 2

                # Save the updated velocity.
                return (array_ops.identity(covariance.assign(new_covariance)), var)

        # Go through variable and update its associated part of the velocity vector.
        return [_update_covariance(vec, var) for vec, var in vecs_and_vars]
