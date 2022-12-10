import tensorflow as tf
from preprocess import *
import math 

def batch_decode(outputs, tokenizer, use_bos, reverse, reverse_last_line):
    """
    Outputs list of python tensors. 
    tokenizer - tokenizer
    use_bos = whether use this token
    reverse = wehther tokens in reversed order. 
    """
    if reverse is True:
        reversed = []
        for output in outputs:
            reversedLine = reverseLineOrder(input_ids = output.numpy(), use_bos = use_bos, tokenizer = tokenizer, reverse_last_line = reverse_last_line)
            #i think -1 flattens it not sure. 
            
            output = tf.reshape(tf.convert_to_tensor(reversedLine), shape = -1)
            print("output shape: ", output.shape)
            reversed.append(output)
        outputs = tf.concat(reversed, axis=0)
        assert(outputs.shape == (len(reversed), output.shape[-1]))
    else:
        outputs = tf.concat(outputs, axis=0)
    
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens = False)
    return outputs
def count_lines(prompt):
    return len(prompt.strip().split("\n"))

def lengths_to_mask(lengths):
    max_len = tf.math.reduce_max(lengths)
    numLengths = lengths.shape[0]

    paddedArray = np.ones(shape = (numLengths, max_len))
    for length in lengths:
        sizePad = max_len - length
        paddedArray[:sizePad] = 0
    return tf.convert_to_tensor(paddedArray, tf.int32)
    


def get_input_ids(prompt, tokenizer, use_bos, reverse, add_line_token):
    prompt = prompt.strip()
    if add_line_token:
        if prompt != "" and prompt[-6:] !="<LINE>":
            prompt += " <LINE>"
    if use_bos and prompt[:5] != "<BOS>":
        prompt = "<BOS> " + prompt
    if reverse is True:
        input_ids = reverseLineOrder(input_ids = tokenizer(prompt, return_tensors = "np").input_ids[0], use_bos = use_bos, tokenizer=tokenizer, reverse_last_line = True)
        input_ids = tf.reshape(tf.constant(prompt), (1,-1))
    else:
        input_ids = tokenizer(prompt, return_tensors = "tf")
    return input_ids["input_ids"]

def generate_both(model, tokenizer,  min_length, max_length, use_bos, reverse, prompts, num_generation, batch_size, add_line_token, lines,  bos_token_id, eos_token_id, pad_token_id):
    
    """
    Generate/finish one line of the poem. Prompts should be in correct word order. 
    """

    """
    Step 1: Concat input ids into large tensor. Prompts are variable length, need to pad BEFORE
    the prompt, generate attention mask. 
    """
    full_input_ids = []
    num_lines = []
    for prompt in prompts:
        if prompt is None:
            prompt = ""
        num_lines = count_lines(prompt)
        #input_ids = get_input_ids(prompt, tokenizer, use_bos, reverse, add_line_token)
        input_ids= get_input_ids(prompt, tokenizer, use_bos, reverse, add_line_token)
        print("input ids: ", input_ids)
        print("id shape: ", input_ids.shape)
        input_ids = tf.tile(input_ids, (num_generation, 1))
        full_input_ids.append(input_ids)

    lengths = []
    for input_ids in full_input_ids:
         lengths += [input_ids.shape[1]] * input_ids.shape[0]
    lengths = tf.convert_to_tensor(lengths, dtype = tf.int32)

    full_attention_mask = lengths_to_mask(lengths)

    #pad input ids:
    max_seq_len = max([input_ids.shape[1] for input_ids in full_input_ids])
    print("full input ids: ", full_input_ids)
    full_input_ids = tf.concat([tf.concat([tf.fill((id.shape[0], max_seq_len - id.shape[1]), value = tokenizer.eos_token_id), id], axis=1) for id in full_input_ids], axis=0)
    print("full input i ds: ", full_input_ids) 
    
    num_batches = math.ceil(full_input_ids.shape[0]/batch_size)
       
    
    #STEP 2: go through each batch and get the gneeration output. 
    outputs = []
    for i in range(num_batches):
        #get a batch of input ids. 
        input_ids = full_input_ids[i*batch_size: (i+1*batch_size)]
        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = full_attention_mask[i*batch_size: (i+1)*batch_size]
        attention_mask = tf.cast(attention_mask, tf.int32)
       
        output = model.generate(input_ids, min_length, max_length, bos_token_id, eos_token_id, pad_token_id)
        
        outputs.append(output)
    print("outputs here: ", outputs)
    #STEP 3: Convert generated result to strings: 
    outputs = batch_decode(outputs, tokenizer, use_bos, reverse, False)
    return outputs

            

def generate_lines(model, tokenizer, min_length, max_length, use_bos, reverse, prompts, num_generation, batch_size, add_line_token,  bos_token_id, eos_token_id, pad_token_id):
    return generate_both(model, tokenizer, min_length, max_length,  use_bos, reverse, prompts, num_generation, batch_size, add_line_token, lines = True,  bos_token_id = bos_token_id, eos_token_id = eos_token_id, pad_token_id = pad_token_id)
def generate_poems(model, tokenizer, device, use_bos, reverse, order, prompts, generate_params, num_generation, batch_size, add_line_token):
        return generate_both(model,tokenizer, device, use_bos, reverse, order, prompts, generate_params, num_generation, batch_size, add_line_token, False)
def generate_new_lines(model, tokenizer, config, prompts, generate_params, num_generation, batch_size):
    return generate_lines(model, tokenizer, config, prompts, generate_params, num_generation, batch_size, True)
def finish_lines(model, tokenizer, config, prompts, generate_params, num_generation, batch_size):
    return generate_lines(model, tokenizer, config, prompts, generate_params, num_generation, batch_size, False)
def generate_poems_two_stage(standard_lm, reverse_lm, standard_tokenizer, reverse_tokenizer, standard_config, reverse_config, prompts, generate_params, num_generation_1 = 10, num_generation_2 = 1, batch_size = 64):
    first_lines = finish_lines(model = standard_lm, tokenizer = standard_tokenizer,config = standard_config , prompts = prompts, generate_params = generate_params, num_generation = num_generation_1, batch_size = batch_size)
    poems = generate_poems(model = reverse_lm, tokenizer = reverse_tokenizer, config = reverse_config, prompts = first_lines, generate_params = generate_params, num_generation = num_generation_2, batch_size = batch_size)
    return poems
def get_last_words(prompt):
    prompt = prompt.split()
    return
def pad_tokens(tokens, tokenizer, max_len):
    return
def get_rhyming_word_score(reverse_lm, tokenizer, config, prompts, rhymes, temperature, batch_size = 64):
    return
def generate_poems_two_stage_with_rhyming(standard_lm, reverse_lm, standard_tokenizer, reverse_tokenizer, standard_config, reverse_config, prompts, generate_params, weighted, num_generation_1 = 10, num_generation_2 = 1, batch_size = 1):
     lines = finish_lines(
        model=standard_lm,
        tokenizer=standard_tokenizer,
        config=standard_config,
        prompts=prompts,
        generate_params=generate_params,
        num_generation=num_generation_1,
        batch_size=batch_size)
     for _ in range(4):
        lines = attach_next_rhyming_word(
            reverse_lm=reverse_lm,
            tokenizer=reverse_tokenizer,
            config=reverse_config,
            prompts=lines,
            num_samples=1,
            weighted=weighted,
            temperature=1.0)
        lines = finish_lines(
            model=reverse_lm,
            tokenizer=reverse_tokenizer,
            config=reverse_config,
            prompts=lines,
            generate_params=generate_params,
            num_generation=1,
            batch_size=batch_size)
     return lines
