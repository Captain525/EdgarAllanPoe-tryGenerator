from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
class GPT2(tf.keras.Model):
    def __init__(self, vocab_size, nPositions, embeddingSize, nLayers, nHead, nInner, tokenizer):
        super().__init__()
        self.config = GPT2Config(vocab_size, n_positions = nPositions, n_embd = embeddingSize, n_layer = nLayers, n_head = nHead, n_inner = nInner, activation_function = "gelu")
        self.pretrained = TFGPT2Model(config = self.config, trainable = False).from_pretrained("gpt2")
        print("tokenizer length: ", len(tokenizer))
        self.pretrained.resize_token_embeddings(len(tokenizer))

        print("vocab size: ", self.pretrained.transformer.vocab_size)
        print(" num positions: ", self.pretrained.config.n_positions )
        #self.dense = tf.keras.layers.Dense(vocab_size, activation = "softmax")

    def call(self, inputs, training):
        #need input shape to be: inputIDs, past, attention)mask, token_type_ids, position_ids, head_mask, \
        poems = inputs[0]
        print(inputs)
        print(poems)
        mask = inputs[1]
        labels = inputs[2]
        print("mask is: ", mask)
        labels = mask*labels
        #print(tf.max(labels))
        pretrainedDict = self.pretrained(input_ids= tf.cast(poems, tf.int32), attention_mask = mask)
        print(pretrainedDict)
       
        return pretrainedDict
    def summary(self):
        pretrained =self.layers[0]
        print("pretrained layers", pretrained.layers[0].h)
        #for layer in self.layers:
            #print(layer)
        #print(self.dense)


    def batch_step(self, inputs, training):
        #extra tuple on outside for some reason. 
        inputs = inputs[0]
        with tf.GradientTape() as tape: 
            logits = self(inputs)
            loss = self.compute_loss(logits, inputs)
        if training:
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss,logits
    def train_step(self, inputs):
        return self.batch_step(inputs, True)
    def test_step(self, inputs):
        return self.batch_step(inputs,False)