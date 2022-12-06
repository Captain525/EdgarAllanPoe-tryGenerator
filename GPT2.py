from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf
class GPT2(tf.keras.Model):
    def __init__(self, vocab_size, n_positions, embeddingSize, nLayers, nHead, nInner):
        super().__init__()
        self.config = GPT2Config(vocab_size, max_position_embeddings = n_positions, n_positions = n_positions, n_embd = embeddingSize, n_layer = nLayers, n_head = nHead, n_inner = nInner, activation_function = "gelu")
        self.pretrained = TFGPT2LMHeadModel(config = self.config).from_pretrained('gpt2')
        #self.dense = tf.keras.layers.Dense(vocab_size, activation = "softmax")

    def call(self, inputs, training):
        poems = inputs[0]
        labels = inputs[1]
     
        pretrainedDict = self.pretrained((tf.cast(poems, tf.int32), labels = labels, position_ids = poems.shape[1], ))
        loss, logits = pretrainedDict['loss'], pretrainedDict['logits']
       
        return loss,logits
    def summary(self):
        pretrained =self.layers[0]
        print("pretrained layers", pretrained.layers[0].h)
        #for layer in self.layers:
            #print(layer)
        #print(self.dense)


    def batch_step(self, inputs, training):
        poemBatch = inputs[0]
        labels = inputs[1]

        with tf.GradientTape() as tape: 
            loss, logits = self((poemBatch, labels))
        if training:
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss,logits
    def train_step(self, inputs):
        return self.batch_step(inputs, True)
    def test_step(self, inputs):
        return self.batch_step(inputs,False)