# copied from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
import logging
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


class ScoreActor(nn.Module):
    time_preprocess: nn.Module
    cond_encoder: nn.Module
    reverse_network: nn.Module

    def __call__(self, obs_enc, actions, time, lora_params, train=False):
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff, train=train)
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug(
                "Broadcasting obs_enc from %s to %s", obs_enc.shape, new_shape
            )
            obs_enc = jnp.broadcast_to(obs_enc, new_shape)

        reverse_input = jnp.concatenate([cond_enc, obs_enc, actions], axis=-1)
        eps_pred = self.reverse_network(reverse_input, lora_params, train=train)
        return eps_pred


class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jax.Array):
        if self.learnable:
            w = self.param(
                "kernel",
                nn.initializers.normal(0.2),
                (self.output_size // 2, x.shape[-1]),
                jnp.float32,
            )
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    hypernet_kwargs: dict
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, lora_params, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)

        x_origin = nn.Dense(self.features * 4)(x)
        if self.hypernet_kwargs.get('diffusion_lora', False):
            lora_A = lora_params['diffusion_residual_kernel_0_lora_A'].reshape(lora_params['diffusion_residual_kernel_0_lora_A'].shape[0], -1, self.hypernet_kwargs["lora_rank"])
            lora_B = lora_params['diffusion_residual_kernel_0_lora_B'].reshape(lora_params['diffusion_residual_kernel_0_lora_B'].shape[0], self.hypernet_kwargs["lora_rank"], -1)
            lora_x = (x @ lora_A @ lora_B) * self.hypernet_kwargs["lora_alpha"] / self.hypernet_kwargs["lora_rank"]
            x = x_origin + lora_x
        else:
            x = x_origin

        x = self.act(x)

        x_origin = nn.Dense(self.features)(x)
        if self.hypernet_kwargs.get('diffusion_lora', False):
            lora_A = lora_params['diffusion_residual_kernel_1_lora_A'].reshape(lora_params['diffusion_residual_kernel_1_lora_A'].shape[0], -1, self.hypernet_kwargs["lora_rank"])
            lora_B = lora_params['diffusion_residual_kernel_1_lora_B'].reshape(lora_params['diffusion_residual_kernel_1_lora_B'].shape[0], self.hypernet_kwargs["lora_rank"], -1)
            lora_x = (x @ lora_A @ lora_B) * self.hypernet_kwargs["lora_alpha"] / self.hypernet_kwargs["lora_rank"]
            x = x_origin + lora_x
        else:
            x = x_origin

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    hypernet_kwargs: dict
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jax.typing.ArrayLike, lora_params, train: bool = False) -> jax.Array:

        x_origin = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        if self.hypernet_kwargs.get('diffusion_lora', False):
            lora_A = lora_params['diffusion_input_lora_A'].reshape(lora_params['diffusion_input_lora_A'].shape[0], -1, self.hypernet_kwargs["lora_rank"])
            lora_B = lora_params['diffusion_input_lora_B'].reshape(lora_params['diffusion_input_lora_B'].shape[0], self.hypernet_kwargs["lora_rank"], -1)
            lora_x = (x @ lora_A @ lora_B) * self.hypernet_kwargs["lora_alpha"] / self.hypernet_kwargs["lora_rank"]
            x = x_origin + lora_x
        else:
            x = x_origin

        for i in range(self.num_blocks):
            if self.hypernet_kwargs.get('diffusion_lora', False):
                lora_residual_params = {
                    'diffusion_residual_kernel_0_lora_A': lora_params['diffusion_residual_kernel_0_lora_A'][i], 
                    'diffusion_residual_kernel_0_lora_B': lora_params['diffusion_residual_kernel_0_lora_B'][i], 
                    'diffusion_residual_kernel_1_lora_A': lora_params['diffusion_residual_kernel_1_lora_A'][i], 
                    'diffusion_residual_kernel_1_lora_B': lora_params['diffusion_residual_kernel_1_lora_B'][i], 
                }
            else:
                lora_residual_params = None
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activation,
                hypernet_kwargs=self.hypernet_kwargs,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, lora_residual_params, train=train)

        x = self.activation(x)

        x_origin = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if self.hypernet_kwargs.get('diffusion_lora', False):
            lora_A = lora_params['diffusion_output_lora_A'].reshape(lora_params['diffusion_output_lora_A'].shape[0], -1, self.hypernet_kwargs["lora_rank"])
            lora_B = lora_params['diffusion_output_lora_B'].reshape(lora_params['diffusion_output_lora_B'].shape[0], self.hypernet_kwargs["lora_rank"], -1)
            lora_x = (x @ lora_A @ lora_B) * self.hypernet_kwargs["lora_alpha"] / self.hypernet_kwargs["lora_rank"]
            x = x_origin + lora_x
        else:
            x = x_origin

        return x


def create_diffusion_model(
    out_dim: int,
    time_dim: int,
    num_blocks: int,
    hypernet_kwargs: dict,
    dropout_rate: float,
    hidden_dim: int,
    use_layer_norm: bool,
):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),
        MLP((2 * time_dim, time_dim)),
        MLPResNet(
            num_blocks,
            out_dim,
            hypernet_kwargs,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )
