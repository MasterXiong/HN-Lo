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
        batch_size, instruction_token_len = task_tokens.shape[0], task_tokens.shape[1]
        # projection layer for the task tokens
        task_tokens = nn.Dense(
            self.hypernet_kwargs["context_embedding_dim"], 
            name="task_token_projection"
        )(task_tokens)
        # add PE to task tokens
        task_tokens += self._create_positional_embedding('instruction', task_tokens)
        # TF layer tokens
        if self.hypernet_kwargs.get('separate_token_for_base_layers', True):
            # a separate token for each Transformer layer in the base network
            base_token_num = layer_token_num = self.base_model_kwargs["num_layers"]
            TF_layer_tokens = jnp.zeros((task_tokens.shape[0], layer_token_num, self.hypernet_kwargs["context_embedding_dim"]))
            TF_layer_tokens += self._create_positional_embedding('TF_layer', TF_layer_tokens)
            # a separate token for each diffusion layer in the base network
            if self.hypernet_kwargs.get('diffusion_lora', False):
                # diffusion reverse network layer tokens
                diffusion_layer_tokens = jnp.zeros((task_tokens.shape[0], 5, self.hypernet_kwargs["context_embedding_dim"]))
                diffusion_layer_tokens += self._create_positional_embedding('diffusion_layer', diffusion_layer_tokens)
                # context input to context encoder
                context_tokens = jnp.concatenate([task_tokens, TF_layer_tokens, diffusion_layer_tokens], axis=1)
                base_token_num += 5
            else:
                context_tokens = jnp.concatenate([task_tokens, TF_layer_tokens], axis=1)
        else:
            base_network_token = jnp.zeros((task_tokens.shape[0], 1, self.hypernet_kwargs["context_embedding_dim"]))
            base_network_token += self._create_positional_embedding('whole_base_network', base_network_token)
            context_tokens = jnp.concatenate([task_tokens, base_network_token], axis=1)
            base_token_num = 1

        # define attention mask: each row of the mask determines how a token attends to the other tokens
        # 1. determine if padding tokens in the instruction will be attended to (yes by default)
        if self.hypernet_kwargs["attend_to_padding"]:
            instruction_attention_mask = jnp.ones((batch_size, 1, context_tokens.shape[-2], instruction_token_len), dtype=bool)
        else:
            instruction_attention_mask = jnp.broadcast_to(jnp.expand_dims(token_mask, (1, 2)), (batch_size, 1, context_tokens.shape[-2], instruction_token_len)).astype(bool)
        # 2. determine if task tokens attend to layer tokens (no by default)
        layer_attention_mask = jnp.ones((batch_size, 1, context_tokens.shape[-2], base_token_num), dtype=bool)
        if not self.hypernet_kwargs["task_attend_to_layer"]:
            layer_attention_mask = layer_attention_mask.at[:, :, :instruction_token_len, :].set(False)
        if not self.hypernet_kwargs.get('layer_token_self_attention', True):
            layer_attention_mask = layer_attention_mask.at[:, :, instruction_token_len:, :].set(False)
        # concat
        attention_mask = jnp.concatenate((instruction_attention_mask, layer_attention_mask), axis=-1)

        # context encoder by transformer
        output = Transformer(**self.hypernet_kwargs["context_encoder_kwargs"])(
            context_tokens, attention_mask, train=train
        )

        if self.hypernet_kwargs.get('separate_token_for_base_layers', True):
            # get the context embedding for each TF layer
            TF_context_embedding = output[:, 16:(16 + layer_token_num)]
            # transpose to shape: layer_num * batch_size * context_embedding_dim
            TF_context_embedding = jnp.transpose(TF_context_embedding, (1, 0, 2))
            # apply dropout to the final context embedding
            embedding_dropout_rate = self.hypernet_kwargs.get("embedding_dropout_rate", 0.)
            TF_context_embedding = nn.Dropout(rate=embedding_dropout_rate)(TF_context_embedding, deterministic=not train)
        else:
            base_network_context_embedding = jnp.repeat(jnp.expand_dims(output[:, -1], axis=0), self.base_model_kwargs["num_layers"], axis=0)
            TF_positional_embedding = self.param(
                "TF_pos_embedding",
                nn.initializers.normal(stddev=0.02),
                (self.base_model_kwargs["num_layers"], 1, self.hypernet_kwargs["context_embedding_dim"]), # layer_num * batch_size * context_embedding_dim
            )
            TF_context_embedding = base_network_context_embedding + TF_positional_embedding
            TF_context_embedding = nn.Dense(
                self.hypernet_kwargs["context_embedding_dim"],
                kernel_init=nn.initializers.normal(stddev=1e-6),
                bias_init=nn.initializers.normal(stddev=1e-6),
                name='TF_context_encoder',
            )(TF_context_embedding)
            TF_context_embedding = nn.relu(TF_context_embedding)

        # HN output heads
        lora_params = dict()
        # initialize matrix A following bias-init
        if self.hypernet_kwargs.get('separate_token_for_lora_module', False):
            MLP_0_lora_A_embedding = TF_context_embedding[:self.base_model_kwargs["num_layers"]]
            MLP_0_lora_B_embedding = TF_context_embedding[self.base_model_kwargs["num_layers"]:(2 * self.base_model_kwargs["num_layers"])]
            MLP_1_lora_A_embedding = TF_context_embedding[(2 * self.base_model_kwargs["num_layers"]):(3 * self.base_model_kwargs["num_layers"])]
            MLP_1_lora_B_embedding = TF_context_embedding[(3 * self.base_model_kwargs["num_layers"]):]
        else:
            MLP_0_lora_A_embedding = TF_context_embedding
            MLP_0_lora_B_embedding = TF_context_embedding
            MLP_1_lora_A_embedding = TF_context_embedding
            MLP_1_lora_B_embedding = TF_context_embedding

        lora_params['MLP_0_lora_A'] = nn.Dense(
            self.token_embedding_size * self.hypernet_kwargs["lora_rank"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.normal(stddev=1e-6),
            name='hypernet_head_MLP_0_lora_A',
        )(MLP_0_lora_A_embedding)
        # initialize matrix B as 0 following LoRA paper
        lora_params['MLP_0_lora_B'] = nn.Dense(
            self.hypernet_kwargs["lora_rank"] * self.base_model_kwargs["mlp_dim"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='hypernet_head_MLP_0_lora_B',
        )(MLP_0_lora_B_embedding)
        lora_params['MLP_1_lora_A'] = nn.Dense(
            self.base_model_kwargs["mlp_dim"] * self.hypernet_kwargs["lora_rank"],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.normal(stddev=1e-6),
            name='hypernet_head_MLP_1_lora_A',
        )(MLP_1_lora_A_embedding)
        lora_params['MLP_1_lora_B'] = nn.Dense(
            self.hypernet_kwargs["lora_rank"] * self.token_embedding_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name='hypernet_head_MLP_1_lora_B',
        )(MLP_1_lora_B_embedding)

        if self.hypernet_kwargs.get('diffusion_lora', False):
            if self.hypernet_kwargs.get('separate_token_for_base_layers', True):
                # get the context embedding for each diffusion layer
                diffusion_context_embedding = output[:, -5:]
                # transpose to shape: layer_num * batch_size * context_embedding_dim
                diffusion_context_embedding = jnp.transpose(diffusion_context_embedding, (1, 0, 2))
            else:
                base_network_context_embedding = jnp.repeat(jnp.expand_dims(output[:, -1], axis=0), 5, axis=0)
                diffusion_positional_embedding = self.param(
                    "diffusion_pos_embedding",
                    nn.initializers.normal(stddev=0.02),
                    (5, 1, self.hypernet_kwargs["context_embedding_dim"]), # layer_num * batch_size * context_embedding_dim
                )
                diffusion_context_embedding = base_network_context_embedding + diffusion_positional_embedding
                diffusion_context_embedding = nn.Dense(
                    self.hypernet_kwargs["context_embedding_dim"],
                    kernel_init=nn.initializers.normal(stddev=1e-6),
                    bias_init=nn.initializers.normal(stddev=1e-6),
                    name='diffusion_context_encoder',
                )(diffusion_context_embedding)
                diffusion_context_embedding = nn.relu(diffusion_context_embedding)
            # input layer for diffusion
            lora_params['diffusion_input_lora_A'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 828,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.normal(stddev=1e-6),
                name='HN_head_for_diffusion_input_lora_A',
            )(diffusion_context_embedding[0])
            lora_params['diffusion_input_lora_B'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 256,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name='HN_head_for_diffusion_input_lora_B',
            )(diffusion_context_embedding[0])
            # residual hidden layers for diffusion
            lora_params['diffusion_residual_kernel_0_lora_A'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 256,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.normal(stddev=1e-6),
                name='HN_head_for_diffusion_residual_kernel_0_lora_A',
            )(diffusion_context_embedding[1:-1])
            lora_params['diffusion_residual_kernel_0_lora_B'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 1024,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name='HN_head_for_diffusion_residual_kernel_0_lora_B',
            )(diffusion_context_embedding[1:-1])
            lora_params['diffusion_residual_kernel_1_lora_A'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 1024,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.normal(stddev=1e-6),
                name='HN_head_for_diffusion_residual_kernel_1_lora_A',
            )(diffusion_context_embedding[1:-1])
            lora_params['diffusion_residual_kernel_1_lora_B'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 256,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name='HN_head_for_diffusion_residual_kernel_1_lora_B',
            )(diffusion_context_embedding[1:-1])
            # output layer for diffusion
            lora_params['diffusion_output_lora_A'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 256,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.normal(stddev=1e-6),
                name='HN_head_for_diffusion_output_lora_A',
            )(diffusion_context_embedding[-1])
            lora_params['diffusion_output_lora_B'] = nn.Dense(
                self.hypernet_kwargs["lora_rank"] * 28,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name='HN_head_for_diffusion_output_lora_B',
            )(diffusion_context_embedding[-1])

        return lora_params


    def _create_positional_embedding(self, name: str, tokens: jax.Array):
        shape = (1, *tokens.shape[-2:])
        embedding = self.param(
            f"{name}_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            shape,
        )
        return jnp.broadcast_to(embedding, tokens.shape)
