import jax
import jax.numpy as jnp
from flax.core.frozen_dict import unfreeze, freeze
from jax_llama import convert_llama_weights, LLaMA, FlaxLLaMAForCausalLM, get_llama_param_partition_spec, LLaMA2Tokenizer, LLaMA3Tokenizer
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import fire

def load(ckpt_dir: str, tokenizer_path: str, is_llama3: bool, **model_kwargs) -> LLaMA:
    # setup jax mesh
    devices = mesh_utils.create_device_mesh((1, len(jax.devices())))
    mesh = Mesh(devices, axis_names=('dp', 'mp'))
    print(f"Mesh: {mesh}")
    
    # load jax model
    if is_llama3:
        tokenizer = LLaMA3Tokenizer(tokenizer_path)
    else:
        tokenizer = LLaMA2Tokenizer(tokenizer_path)
    jax_params, jax_config = convert_llama_weights(ckpt_dir, tokenizer)
    with jax.default_device(jax.devices('cpu')[0]):
        jax_params = freeze(jax.tree_map(lambda x: jnp.asarray(x), jax_params))
    # shard params
    param_spec = freeze(get_llama_param_partition_spec(unfreeze(jax_params), fsdp=False))
    jax_params = jax.tree_util.tree_map(lambda param, spec: jax.device_put(param, NamedSharding(mesh, spec)), jax_params, param_spec)

    # build model
    jax_model = FlaxLLaMAForCausalLM(jax_config, _do_init=False, **model_kwargs)

    return LLaMA(jax_params, jax_model, tokenizer, mesh=mesh)

def main(ckpt_dir: str, tokenizer_path: str, is_llama3: bool, max_gen_len: int=256, temperature: float = 0.8, top_p: float = 0.95):
    generator = load(ckpt_dir, tokenizer_path, is_llama3)
    prompts = ["The capital of Germany is the city of", "Here is my sonnet in the style of Shakespeare about an artificial intelligence:"]
    results = generator.generate_from_str(prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

    for result in results:
        print(result)
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
