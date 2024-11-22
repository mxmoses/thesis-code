"""NAdam optimizer with reduce LR on Plateau and target awareness in Jax."""
import functools
import logging
from types import SimpleNamespace

# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
from optax._src import base
from optax._src import numerics

from algorithmic_efficiency import spec
import jax.experimental.host_callback as hcb

# Get the experiment name from the environment variable
experiment_name = os.getenv('EXPERIMENT_NAME', 'default_experiment')

# Define the log file path globally
log_file = f'/experiment_runs/{experiment_name}_learning_rate_log.csv'
reduce_lr_log_file = f'/experiment_runs/{experiment_name}_reduce_lr_state_log.csv'

_GRAD_CLIP_EPS = 1e-6

HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0017486387539278373,
    "one_minus_beta1": 0.06733926164,
    "beta2": 0.9955159689799007,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "warmup_factor": 0.02,
    "decay_rate": 0.96,
    "patience": 10,
    "cooldown": 10,
    "factor": 0.5,
    "rtol": 1e-4,
    "accumulation_size": 5
}

class ReduceLROnPlateauState(NamedTuple):
  """State for the ReduceLROnPlateau callback."""

  scale: chex.Array  
  best_value: chex.Array  
  plateau_count: chex.Array  
  cooldown_count: chex.Array  
  count: chex.Array  
  avg_value: chex.Array  

def reduce_on_plateau(
    factor: float = 0.1,
    patience: int = 10,
    rtol: float = 1e-4,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    min_scale: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  
  def init_fn(params) -> ReduceLROnPlateauState:
    del params
    return ReduceLROnPlateauState(
        best_value=jnp.asarray(float("inf"), dtype=jnp.float32),
        plateau_count=jnp.asarray(0, jnp.int32),
        scale=jnp.asarray(1.0, dtype=jnp.float32),
        cooldown_count=jnp.asarray(0, jnp.int32),
        count=jnp.asarray(0, jnp.int32),
        avg_value=jnp.asarray(0.0, jnp.float32),
    )

  def _update_scale(state):
    # Update plateau count and check if plateaued
    avg_value = state.avg_value
    has_improved = jnp.where(
        avg_value < (1 - rtol) * state.best_value - atol, 1, 0
    )
    new_best_value = jnp.where(
        has_improved, avg_value, state.best_value
    )
    curr_plateau_count = jnp.where(
        has_improved, 0, numerics.safe_int32_increment(state.plateau_count)
    )

    # We're in cooldown, so reduce the counter and ignore any bad epochs
    def in_cooldown():
      new_plateau_count = jnp.asarray(0, jnp.int32)
      new_scale = state.scale
      new_cooldown_count = state.cooldown_count - 1
      return new_plateau_count, new_scale, new_cooldown_count

    # We're not in cooldown, so update the plateau count and scale as usual
    def not_in_cooldown():
      new_plateau_count = jnp.where(
          curr_plateau_count == patience, 0, curr_plateau_count
      )
      new_scale = jnp.maximum(
          jnp.where(
              curr_plateau_count == patience,
              state.scale * factor,
              state.scale,
          ),
          min_scale,
      )
      new_cooldown_count = jnp.where(
          curr_plateau_count == patience, cooldown, 0
      )
      new_cooldown_count = jnp.asarray(new_cooldown_count, dtype=jnp.int32)


      return new_plateau_count, new_scale, new_cooldown_count

    new_plateau_count, new_scale, new_cooldown_count = jax.lax.cond(
        state.cooldown_count > 0, in_cooldown, not_in_cooldown
    )
    new_state = ReduceLROnPlateauState(
        plateau_count=new_plateau_count,
        best_value=new_best_value,
        scale=new_scale,
        cooldown_count=new_cooldown_count,
        count=jnp.asarray(0, dtype=jnp.int32),
        avg_value=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return new_state

  def update_fn(
      updates: base.Updates,
      state: ReduceLROnPlateauState,
      params=None,
      *,
      value: float,
      **extra_args,
  ) -> tuple[base.Params, ReduceLROnPlateauState]:
    del params, extra_args

    count = state.count
    new_count = numerics.safe_int32_increment(count)
    new_avg_value = (
        count * state.avg_value + jnp.asarray(value, dtype=state.avg_value.dtype)
    ) / new_count
    new_state = state._replace(avg_value=new_avg_value, count=new_count)

    new_state = jax.lax.cond(
        new_count == accumulation_size, _update_scale, lambda x: x, new_state
    )

    updates = jax.tree_util.tree_map(lambda g: new_state.scale * g, updates)

    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)

global_lr_schedule_fn = None
lr_reset_done = False


def log_learning_rate_callback(arg, transforms):
    global_step, learning_rate = arg
    if jax.process_index() == 0:  # Log only on the first device
        
        try:
            # Ensure scalar conversion
            global_step_scalar = int(jnp.asarray(global_step).item())
            learning_rate_scalar = float(jnp.asarray(learning_rate)[0])  # Index to get the scalar value
            
            log_learning_rate(global_step_scalar, learning_rate_scalar, log_file)
        
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded
                print(f"Warning: Disk quota exceeded. Stopping logging at step {global_step_scalar}.")
            else:
                print(f"Error: {e.strerror}. Stopping logging at step {global_step_scalar}.")
        
        except Exception as e:
            print(f"Unexpected error occurred: {e}. Stopping logging at step {global_step_scalar}.")
        
def log_learning_rate(global_step, learning_rate, log_file):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([global_step, learning_rate])

def initialize_log_file(log_file, headers):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def log_reduce_lr_state(global_step, scale, best_value, plateau_count, cooldown_count, log_file):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([global_step, scale, best_value, plateau_count, cooldown_count])

initialize_log_file(reduce_lr_log_file, ['Global Step', 'Scale', 'Best Value', 'Plateau Count', 'Cooldown Count'])


# Forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/alias.py
def nadam(
    learning_rate: Union[float, optax.Schedule],
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    hyperparameters: dict = None
) -> optax.GradientTransformation:
  
  # Chains two transformations: scale_by_nadam and scale_by_learning_rate.
  # scale_by_nadam configures the gradient transformation according to the NAdam algorithm's logic.
  # scale_by_learning_rate modifies the learning rate as specified, either statically or according to a schedule.
  if hyperparameters is None:
      hyperparameters = {}
  patience = getattr(hyperparameters, 'patience', 5)
  cooldown = getattr(hyperparameters, 'cooldown', 0)
  factor = getattr(hyperparameters, 'factor', 0.5)
  rtol = getattr(hyperparameters, 'rtol', 1e-4)
  accumulation_size = getattr(hyperparameters, 'accumulation_size', 200)
  
  return optax.chain(
      scale_by_nadam(b1, b2, eps, eps_root, debias),
      scale_by_learning_rate(learning_rate),
      reduce_on_plateau(
        patience=patience,
        cooldown=cooldown,
        factor=factor,
        rtol=rtol,
        accumulation_size=accumulation_size,
      )
  )

# All functions below are forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/transform.py
def scale_by_nadam(b1: float = 0.9,
                   b2: float = 0.999,
                   eps: float = 1e-8,
                   eps_root: float = 0.0,
                   debias: bool = True,
                   power: float = 0.5) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this).

  Current code implements a simpler version with no momentum decay and slightly
  different (standard Adam) bias correction terms. The exact description can be
  found here https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: Whether to use bias correction.
    power: The power to use in the preconditioner (0.5 in default adam).
  Returns:
    An (init_fn, update_fn) tuple.
  """
  raise_power = jnp.sqrt if power == 0.5 else lambda x: jnp.power(x, power)

  def init_fn(params):
    mu = jax.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = _update_moment(updates, state.mu, b1, 1)
    nu = _update_moment(updates, state.nu, b2, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    mu_hat = _update_moment(updates, mu, b1, 1)
    mu_hat = mu_hat if not debias else _bias_correction(mu_hat, b1, count)
    nu_hat = nu if not debias else _bias_correction(nu, b2, count)
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count, mu=mu, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the NAdam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates


def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_map(
      lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  global global_lr_schedule_fn
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng
  del hyperparameters

  hyperparameters = SimpleNamespace(**HPARAMS)

  # Create optimizer + LR schedule.
  opt_init_fn, opt_update_fn = nadam(
      learning_rate=hyperparameters.learning_rate,
      b1=1.0 - hyperparameters.one_minus_beta1,
      b2=hyperparameters.beta2,
      eps=1e-8)

  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
                       label_smoothing,
                       global_step):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container, value=loss)
  updated_params = optax.apply_updates(current_param_container, updates)

  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, Dict[str, float]]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types, loss_type, hyperparameters
    global lr_reset_done

    hyperparameters = HPARAMS

    # Get performance target and runtime steps
    performance_target = workload.validation_target_value
    max_runtime_in_steps = workload.step_hint

    # Initialize validation_accuracy and assess proximity if eval_results is available
    validation_accuracy = -1
    accuracy_proximity = None
    latest_eval = None
    latest_metrics = {}

    if global_step % 100 == 0 and eval_results:
      if len(eval_results) > 0:
        latest_eval = eval_results[-1]
        _, latest_metrics = latest_eval
        validation_accuracy = latest_metrics.get('validation/accuracy', -1)
      else:
        print(f"Global Step {global_step}: No evaluation results available.")

    del eval_results

    optimizer_state, opt_update_fn = optimizer_state
    per_device_rngs = jax.random.split(rng, jax.local_device_count())
    label_smoothing = getattr(hyperparameters, 'label_smoothing', 0.0)
    grad_clip = getattr(hyperparameters, 'grad_clip', None)
    
    outputs = pmapped_train_step(workload,
                                 opt_update_fn,
                                 model_state,
                                 optimizer_state,
                                 current_param_container,
                                 batch,
                                 per_device_rngs,
                                 grad_clip,
                                 label_smoothing,
                                 global_step)
    new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

    new_optimizer_state_host = jax.device_get(new_optimizer_state)
    reduce_lr_on_plateau_state = new_optimizer_state_host[2]

    # Single adjustment logic
    step_fraction = 0.35  # Trigger adjustment after 35% of runtime steps
    distance_threshold = 0.01  # Trigger adjustment if far from target

    # Log loss and grad_norm
    if global_step % 100 == 0 and workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
            {
                'loss': loss[0],
                'grad_norm': grad_norm[0],
            }, global_step)
        if not lr_reset_done and global_step > max_runtime_in_steps * step_fraction and validation_accuracy != -1:
          distance_from_target = abs(performance_target - validation_accuracy)
          if distance_from_target > distance_threshold:
              print(f"Global Step {global_step}: Far from target after {step_fraction * 100}% of runtime steps. Resetting LR.")

              # Reset scale to higher value and ensure correct shape
              modified_scale = (reduce_lr_on_plateau_state.scale * 2.5).reshape(-1)  # Reshape to ensure the correct dimension
              reduce_lr_on_plateau_state = reduce_lr_on_plateau_state._replace(
                  scale=modified_scale
              )

              new_optimizer_state_host = (
                  new_optimizer_state_host[0],
                  new_optimizer_state_host[1],
                  reduce_lr_on_plateau_state
              )
              lr_reset_done = True  # Mark reset as done
          else:
              print(f"Global Step {global_step}: Close enough to target, no adjustment needed.")
          
    # Finalize optimizer state update
    new_optimizer_state = jax.device_put(new_optimizer_state_host)

    # Log ReduceLR State Information
    best_value = reduce_lr_on_plateau_state.best_value
    plateau_count = reduce_lr_on_plateau_state.plateau_count
    cooldown_count = reduce_lr_on_plateau_state.cooldown_count
    scale = reduce_lr_on_plateau_state.scale

    hcb.id_tap(log_learning_rate_callback, (global_step, hyperparameters['learning_rate'] * reduce_lr_on_plateau_state.scale))
    log_reduce_lr_state(global_step, scale, best_value, plateau_count, cooldown_count, reduce_lr_log_file)

    return (new_optimizer_state, opt_update_fn), new_params, new_model_state



def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'cifar':
    batch_size = 32
  elif workload_name == 'criteo1tb':
    batch_size = 262_144
  elif workload_name == 'fastmri':
    batch_size = 16
  elif workload_name == 'imagenet_resnet':
    batch_size = 64
  elif workload_name == 'imagenet_vit':
    batch_size = 1024
  elif workload_name == 'librispeech_conformer':
    batch_size = 224
  elif workload_name == 'librispeech_deepspeech':
    batch_size = 128
  elif workload_name == 'mnist':
    batch_size = 16
  elif workload_name == 'ogbg':
    batch_size = 256
  elif workload_name == 'wmt':
    batch_size = 64
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')
  
  logging.info(f'Batch size for {workload_name}: {batch_size}')
  return batch_size


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch