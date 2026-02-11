"""LLaDA2 MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig


class LLaDA2MoeConfig(PretrainedConfig):
    model_type = "llada2_moe"

    def __init__(
        self,
        vocab_size:int=30592,
        hidden_size:int=1024,
        intermediate_size:int=None,
        num_hidden_layers:int=24,
        num_attention_heads:int=16,
        num_key_value_heads:int=0,
        hidden_act:str="silu",
        use_qkv_bias:bool=False,  # llada2 only
        use_qk_norm:bool=True,
        use_bias:bool=True,  # llada2 only
        rms_norm_eps:float=1e-05,
        norm_head:bool=False,  # llada2 only
        tie_word_embeddings:bool=False,  # PretrainedConfig key, here change default value.
        embedding_dropout:float=0.1,
        attention_dropout:float=0.1,
        output_dropout:float=0.1,
        initializer_range:float=0.02,
        max_position_embeddings:int=16384,
        rope_theta:float=10000.0,
        use_cache:bool=True,
        use_sliding_window:bool=False,
        sliding_window:int=4096,
        max_window_layers:int=28,
        rope_scaling=None,
        pad_token_id:int=126081,
        num_experts:int=16,
        num_shared_experts:int=0,
        num_experts_per_tok:int=2,
        n_group:int=8,
        topk_group:int=4,
        routed_scaling_factor:float=2.5,
        moe_intermediate_size:int=None,
        first_k_dense_replace:int=0,
        head_dim:int=None,
        output_router_logits:bool=False,
        partial_rotary_factor:float=0.5,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or self.hidden_size // self.num_attention_heads
        self.rope_scaling = rope_scaling

        # MoE configs
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(
            pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
