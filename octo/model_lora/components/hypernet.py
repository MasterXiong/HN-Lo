from typing import Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from octo.model.components.transformer import Transformer


class Hypernet(nn.Module):
    base_model_kwargs: Dict
    hypernet_kwargs: Dict
    token_embedding_size: int = 768

    @nn.compact
    def __call__(self, task_tokens, token_mask, train: bool):
        '''
        task_tokens: shape = (batch_size * token_num * token_embedding_size)
        '''
        # projection layer for the task tokens
        task_tokens = nn.Dense(
            self.hypernet_kwargs["context_embedding_dim"], 
            name="task_token_projection"
        )(task_tokens)
        # add PE to task tokens
        task_tokens += self._create_positional_embedding('instruction', task_tokens)
        # layer tokens
        layer_tokens = jnp.zeros((task_tokens.shape[0], self.base_model_kwargs["num_layers"], self.hypernet_kwargs["context_embedding_dim"]))
        layer_tokens += self._create_positional_embedding('layer', layer_tokens)
        # context input to context encoder
        context_tokens = jnp.concatenate([task_tokens, layer_tokens], axis=1)
        # define causal attention mask
        # each row of the mask determines how a token attends to the other tokens
        attention_mask = jnp.ones((context_tokens.shape[0], 1, context_tokens.shape[1], context_tokens.shape[1]), dtype=bool)
        attention_mask = attention_mask.at[:, :, :task_tokens.shape[1], task_tokens.shape[1]:].set(False)
        # context encoder by transformer
        # Run transformer
        output = Transformer(**self.hypernet_kwargs["context_encoder_kwargs"])(
            context_tokens, attention_mask, train=train
        )
        # get the context embedding for each layer
        context_embedding = output[:, -self.base_model_kwargs["num_layers"]:]
        # turn into shape: layer_num * batch_size * context_embedding_dim
        context_embedding = jnp.transpose(context_embedding, (1, 0, 2))

        # HN output heads
        lora_params = dict()
        # initialize matrix A following bias-init
        lora_params['MLP_0_lora_A'] = nn.Dense(
            self.token_embedding_size * self.hypernet_kwargs["lora_rank"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.normal(stddev=1e-6),
            name='hypernet_head_MLP_0_lora_A',
        )(context_embedding)
        # initialize matrix B as 0 following LoRA paper
        lora_params['MLP_0_lora_B'] = nn.Dense(
            self.hypernet_kwargs["lora_rank"] * self.base_model_kwargs["mlp_dim"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='hypernet_head_MLP_0_lora_B',
        )(context_embedding)
        lora_params['MLP_1_lora_A'] = nn.Dense(
            self.base_model_kwargs["mlp_dim"] * self.hypernet_kwargs["lora_rank"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.normal(stddev=1e-6),
            name='hypernet_head_MLP_1_lora_A',
        )(context_embedding)
        lora_params['MLP_1_lora_B'] = nn.Dense(
            self.hypernet_kwargs["lora_rank"] * self.token_embedding_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='hypernet_head_MLP_1_lora_B',
        )(context_embedding)

        return lora_params


    def _create_positional_embedding(self, name: str, tokens: jax.Array):
        shape = (1, *tokens.shape[-2:])
        embedding = self.param(
            f"{name}_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            shape,
        )
        return jnp.broadcast_to(embedding, tokens.shape)
