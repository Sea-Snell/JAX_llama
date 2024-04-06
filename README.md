# LLaMA Jax

**UPDATE: now with LLaMA-2 support!**

This repository is a Huggingface compatible port of [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models to Jax.
In order to download the checkpoints and tokenizer for LLaMA, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

### Setup
In a conda env, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

### Download
Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

### Inference
The provided `jax_example.py` will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
python jax_example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

### Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

### License
See the [LICENSE](LICENSE) file.

### Testing
The provided `jax_test.py` script runs a comparison between this jax model and the pytorch version provided by [Meta](https://github.com/facebookresearch/llama). To run the tests, install Meta's code in the same environment and run the script with:

```
torchrun --nproc_per_node MP jax_test.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```
*(Note: some of the tests only run when MP=1)*

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |
