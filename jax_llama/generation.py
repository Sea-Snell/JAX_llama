import jax
import jax.numpy as jnp
from jax_llama import FlaxLLaMAForCausalLM
from jax_llama.llama2_tokenizer import Tokenizer as LLaMA2Tokenizer
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax_llama.partition import with_named_sharding_constraint
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
from flax import struct
from functools import partial
from typing import List, Optional, Union

class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLLaMAForCausalLM = struct.field(pytree_node=False)
    tokenizer: Union[LLaMA2Tokenizer, LLaMA3Tokenizer] = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    @partial(jax.jit, static_argnums=(3,4,5))
    def generate(self, tokens: jnp.ndarray, attention_mask: jnp.ndarray, max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95) -> jnp.ndarray:
        tokens = with_named_sharding_constraint(tokens, self.mesh, P("dp", None))
        attention_mask = with_named_sharding_constraint(attention_mask, self.mesh, P("dp", None))

        generations = self.model.generate(
            input_ids=tokens, 
            attention_mask=attention_mask, 
            params=self.params, 
            generation_config=GenerationConfig(
                num_beams=1, 
                do_sample=temperature != 0.0, 
                max_length=max_gen_len+tokens.shape[1], 
                pad_token_id=self.tokenizer.eos_id, 
                eos_token_id=self.tokenizer.eos_id, 
                temperature=temperature, 
                top_p=top_p, 
            ), 
        )
        out_tokens = generations.sequences
        
        out_tokens = with_named_sharding_constraint(out_tokens, self.mesh, P("dp", None))
        return out_tokens
    
    def generate_from_str(self, prompts: List[str], max_gen_len: int, temperature: float = 0.8, top_p: float = 0.95):
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        max_prompt_size = max([len(t) for t in prompt_tokens])

        tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.eos_id).astype(jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t):].set(t) # left pad
        attention_mask = (tokens != self.tokenizer.eos_id).astype(jnp.int32)

        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p)

        decoded = []
        for i, t in enumerate(out_tokens.tolist()):
            # cut to max gen len
            t = t[t.index(self.tokenizer.bos_id):]
            t = t[:(len(prompt_tokens[i])+max_gen_len)]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
