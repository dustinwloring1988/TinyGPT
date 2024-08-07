## TinyGPT with LLama3 Tokenizer

TinyGPT integrates the GPT-2 model with a LLama3 tokenizer, expanding its vocabulary size to 128K tokens. Some other adjustments include increasing `n_positions` and `n_embd` to 2048.

However, we are encountering inefficiencies in the training code. Our Runpod.io dashboard images reveal that the model is confined to a single GPU rather than distributed across multiple GPUs.

1) What can we do to get ride of one of the GPUS taking up so much more memory than the others.
2) How can we make the code more efficet.

![8 A40](https://raw.githubusercontent.com/dustinwloring1988/TinyGPT/main/dash-of%208%20A40.png)
![4 A40](https://github.com/dustinwloring1988/TinyGPT/blob/main/dash.png)


### TinyGPT Configuration Summary
- **Model Type:** GPT2LMHeadModel
- **Number of Parameters:** 1,072,599,040

#### Model Configuration
- **Vocabulary Size:** 128,256
- **Max Positions:** 2048
- **Embedding Size:** 2048
- **Number of Layers:** 16
- **Number of Attention Heads:** 16

#### Special Tokens
- **BOS Token:** (ID: 128000)
- **EOS Token:** (ID: 128001)
- **PAD Token:** (ID: 128001)
