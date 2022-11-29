#look at this website https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/gpt2#overview


from transformers import GPT2Tokenizer, GPT2Config, TFGPT2Model
vocabSize = 2048
embedSize = 1000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#configuration = GPT2Config(vocab_size = vocabSize, n_positions = 2048, n_embd= embedSize, n_layer = 12, n_head = 12, n_inner = None, activation_function = "relu")

#model with weights
model = TFGPT2Model.from_pretrained('gpt2')