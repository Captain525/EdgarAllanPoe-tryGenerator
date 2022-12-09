import tensorflow as tf
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config, PreTrainedModel
from BeamSearch import generatorHelp
import numpy as np
class GPT2FineTune(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        hidden_size = 100
        self.embedding = tf.keras.layers.Embedding(vocab_size, 768)
        self.embedDense = tf.keras.layers.Dense(768, "relu")
       
        self.transformerBlock = TFGPT2LMHeadModel.from_pretrained("gpt2")
        self.transformerBlock.resize_token_embeddings(vocab_size)
        #want transformerBlock TO NOT TRAIN. TOO MUCH MEMORY REQUIRED, WE WANT TO FINE TUNE IT. 
        self.transformerBlock.trainable = False
   
        self.vocab_size = vocab_size
    
        self.dense = tf.keras.layers.Dense(vocab_size, "softmax")
        #loss function here. 
        self.lossCalc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)

    def compute_loss(self, logits, labels):
        """
        labels is just the input sequence rn. 
        logits are the probabilities(not normalized) given to each one. 
        want to do sparse categorical crossentorpy after shifting labels by 1 and lgoits by 1.

        assuming labels: (bs, sequenceSize)
        assuming logits: (bs, sequenceSize, vocabSize) 
        """
        batchSize = logits.shape[0]
        sequenceSize = logits.shape[1]
        vocabSize = logits.shape[2]

        assert(labels.shape == (batchSize, sequenceSize))
        assert(logits.shape == (batchSize, sequenceSize, vocabSize))
        print("labels before: ", labels)
        labels = labels[:, 1:]
        logits= logits[:,:-1, :]
        print("labels after: " , labels)
        
        #true then predicted. 
        #gets loss as one value instead of as a batch. 
        loss = self.lossCalc(labels, logits)
        
  
        return loss

    def call(self, inputs, training):
        """
        inputs right now is JUST the training ids, no mask or labels. 
        This is shape batchSize x sequenceSize, each element is an tokenized word. 
        """
        bs = inputs.shape[0]
        sequenceSize = inputs.shape[1]

        embeddedInputs = self.embedding(inputs)
        outputs = self.transformerBlock(None, inputs_embeds = embeddedInputs)
        #get just the last hidden state of the outputs. 
        lastHiddenState = outputs[0]
        #print(lastHiddenState.shape)
        #assert(lastHiddenState.shape == (bs, sequenceSize,768))
        #apply dense layer to that hidden state. 
        logits= self.dense(lastHiddenState)

        return logits
    def generate(self, input_ids, max_length, bos_token_id, eos_token_id, pad_token_id):
      
            
        #set the attention mask if there wasn't one already. Set it equal to just being 0 where pad tokens are. 
        
        #generated = self(input_ids)
        generated = self.greedySearch(input_ids, 10, max_length, bos_token_id, eos_token_id, pad_token_id)
        return generated

        
    def greedySearch(self, input_ids, min_length, max_length, bos_token_id,eos_token_id, pad_token_id):
        #output size is equal to the input size so we should just put in one at a time. 
        if input_ids is None: 
            input_ids = tf.fill((1, 1), bos_token_id)
        attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
       
        #generating text AFTER the input sequence. 
        concatSequence= input_ids
        sequenceEnd = tf.ones(shape = (input_ids.shape[0], ), dtype = tf.int32)
        maskBos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == bos_token_id), tf.float32)
        maskEos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == eos_token_id), tf.float32)
        print(tf.range(0, self.vocab_size, 1)[-1])
        print("eos token id: ", eos_token_id)
        maskPad = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == pad_token_id), tf.float32)
        print("Eos pad: ", maskEos)
        assert(maskEos[-1] == 0)
      
        for i in range(0, max_length):
            
            logits = tf.nn.softmax(self.transformerBlock(concatSequence)[0], axis = -1)
         
        
            assert(logits.shape == (concatSequence.shape[0], concatSequence.shape[1], self.vocab_size))
            
            #get the max of the LAST logits. 
            #try batch size of one. 
            log = logits[:, -1]
            log = log*maskBos
            log = log*maskPad
            #make it so can't get the end token if less than minimum length. 
            if i<min_length: 
        
                log = log*maskEos
          
            #if sequence has already ended, just add padding tokens as next guess. If not, (sequenceEnd element = 1), then add the newly chosen element. 
            nextGuess = (tf.argmax(log, axis=-1, output_type = tf.int32)[..., tf.newaxis])*sequenceEnd + pad_token_id*tf.ones(shape = (input_ids.shape[0], 1), dtype = tf.int32)*(not sequenceEnd)
            concatSequence = tf.concat([concatSequence, nextGuess], axis = -1)

            #makes a value 0 (ie already reached end of sequence) if either it was already 0 or the new next guess == eos_token_id. 
            sequenceEnd =tf.cast(tf.math.logical_and((not (nextGuess == eos_token_id)), tf.cast(sequenceEnd, bool)), tf.int32)
        #sequence is size initial sequence + added stuff, basicaly generated new stuff this way. 
        return concatSequence

    def sample(self, input_ids, min_length, max_length, bos_token_id,eos_token_id, pad_token_id):
        if input_ids is None: 
            input_ids = tf.fill((1, 1), bos_token_id)
        attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
       
        #generating text AFTER the input sequence. 
        concatSequence= input_ids
        sequenceEnd = tf.ones(shape = (input_ids.shape[0], ), dtype = tf.int32)
        maskBos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == bos_token_id), tf.float32)
        maskEos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == eos_token_id), tf.float32)
        print(tf.range(0, self.vocab_size, 1)[-1])
        print("eos token id: ", eos_token_id)
        maskPad = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == pad_token_id), tf.float32)
        print("Eos pad: ", maskEos)
        assert(maskEos[-1] == 0)
      
        for i in range(0, max_length):
            
            logits = tf.nn.softmax(self.transformerBlock(concatSequence)[0], axis = -1)
            assert(logits.shape == (concatSequence.shape[0], concatSequence.shape[1], self.vocab_size))
            
            #get the max of the LAST logits. 
            #try batch size of one. 
            log = logits[:, -1]
            log = log*maskBos
            log = log*maskPad
            #make it so can't get the end token if less than minimum length. 
            if i<min_length: 
        
                log = log*maskEos
            randomValues = tf.random.uniform(shape = (concatSequence.shape[0], ))
    def findBin(randomValues, probs):
        vocabSize = probs.shape[-1]
        arrayBin = np.zeros((probs.shape[0],))
        sum = tf.zeros((probs.shape[0],))
        chosen = tf.ones((probs.shape[0], ))
        for i in range(vocabSize):
            newSum = sum + probs[:, i]
            #assuming previous step was greater than. 
            lessThan = randomValues<= newSum
            

    def beamSearch():
        return 
    def batch_step(self, inputs, training):
        """
        """
        with tf.GradientTape() as tape: 
            logits = self(inputs, training)
            #Use inputs as labels, but will shift it to the right by one and shift the other to the left by one. 
            loss = self.compute_loss(logits, inputs)
        if training:
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return {"loss": loss}
    def train_step(self, inputs):
        return self.batch_step(inputs, True)
    def test_step(self, inputs):
        return self.batch_step(inputs, False)
    def resize_token_embeddings(self, size):
        """
        Doesn't really work and allow us to add tokens to the vocabulary. 
        """
        self.transformerBlock.resize_token_embeddings(size)
        print("done")