import tensorflow as tf
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config, PreTrainedModel

import numpy as np
from EvalMetrics import perplexity
class GPT2FineTune(tf.keras.Model):
    def __init__(self, vocab_size, loadWeightsPath=None, modelNum = 0):
        super().__init__()
        if modelNum == 0:
            #EMbedding + GPT + Dense
            self.addTail =True
            self.headModel = False
            self.addEmbedding = True
            self.doMask =True
        if modelNum == 1:
            #embedding + LM GPT
            self.headModel = True
            self.addTail = False
            self.addEmbedding = True
            self.doMask = True
        if modelNum == 2:
            #GPT + Dense
            self.headModel = False
            self.addTail = True
            self.addEmbedding = False
            self.doMask = True
        """
        Before, using sequential model sfor the head and tail caused everything to break, now it doesn't. 

        I think it might be LMHeadModel vs just normal model. LMHeadModel causes it to crash maybe. 
        """

        if self.addEmbedding: 

            self.embedding = tf.keras.layers.Embedding(vocab_size, 768)
            self.head = tf.keras.models.Sequential([self.embedding])
        if self.headModel:
            self.transformerBlock = TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.transformerBlock = TFGPT2Model.from_pretrained("gpt2")
        #want transformerBlock TO NOT TRAIN. TOO MUCH MEMORY REQUIRED, WE WANT TO FINE TUNE IT. 
        self.transformerBlock.trainable = False
       
        self.vocab_size = vocab_size

        if self.addTail:
            self.dense = tf.keras.layers.Dense(vocab_size, "softmax")
            self.tail = tf.keras.models.Sequential([self.dense])
        
        #loss function here. 
        self.lossCalc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
        self.perplexity = tf.keras.metrics.Mean("perplexity")
        if loadWeightsPath is not None: 
            self.load_weights(loadWeightsPath)
    def compute_loss(self, logits, labels, mask):
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
        
        labels = labels[:, 1:]
        logits= logits[:,:-1, :]
        mask = mask[:, 1:]
        
        
        #true then predicted. 
        #gets loss as one value instead of as a batch. 
        loss = self.lossCalc(labels, logits, mask)
        
  
        return loss

    def call(self, inputs, training):
        """
        inputs right now is JUST the training ids, no mask or labels. 
        This is shape batchSize x sequenceSize, each element is an tokenized word. 
        """
        text = inputs[0]
        attention_mask = inputs[1]
        bs = text.shape[0]
        sequenceSize = text.shape[1]
        if self.doMask:
            maskedText = attention_mask * text
        else:
            maskedText = text
            attention_mask = None
        if self.addEmbedding:
            embeddedInputs = self.head(maskedText)
            
            outputs = self.transformerBlock(None, inputs_embeds = embeddedInputs, attention_mask = attention_mask)
        else:
            
            outputs = self.transformerBlock(text, attention_mask = attention_mask)
        #get just the last hidden state of the outputs. 
        lastHiddenState = outputs[0]
        if(self.addTail):
            probs = self.tail(lastHiddenState)
        else:
            probs = tf.nn.softmax(lastHiddenState, axis=-1)
 
        return probs

    def generate(self, input_ids, min_length, max_length, bos_token_id, eos_token_id, pad_token_id):
      
            
        #set the attention mask if there wasn't one already. Set it equal to just being 0 where pad tokens are. 
        

        generated = self.getGeneratedText(input_ids, min_length, max_length, bos_token_id, eos_token_id, pad_token_id, True)
        return generated

    def getGeneratedText(self, input_ids, min_length, max_length, bos_token_id,eos_token_id, pad_token_id, sample):
        """
        Generate a new sequence by sampling randomly from the outputted probability distribution from the model. 
        Iterate through the sequence, generating one new word each time.
        """

        lookAtPrev = True
        if input_ids is None: 
            input_ids = tf.fill((1, 1), bos_token_id)
        attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
        batchSize = input_ids.shape[0]
        #generating text AFTER the input sequence. 
        concatSequence= input_ids
        sequenceEnd = tf.ones(shape = (batchSize, ), dtype = tf.int32)
        maskBos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == bos_token_id), tf.float32)
        if lookAtPrev: 
            previousGuesses = eos_token_id *tf.ones((batchSize, ), dtype = tf.int32)
            comparison = tf.tile(tf.range(0, self.vocab_size, 1)[tf.newaxis, ...], multiples = (batchSize, 1))
            assert(comparison.shape == (batchSize, self.vocab_size))
        assert(maskBos[bos_token_id] == 0)

        maskEos = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == eos_token_id), tf.float32)
        assert(maskEos[eos_token_id] == 0)
      
        
        maskPad = tf.cast(tf.math.logical_not(tf.range(0, self.vocab_size, 1, dtype = tf.int32) == pad_token_id), tf.float32)
       
        assert(maskPad[pad_token_id] == 0)
      
        for i in range(0, max_length):
            attention_mask =tf.cast(tf.math.not_equal(concatSequence, pad_token_id), dtype=tf.int32)
            if lookAtPrev:
                previous_guess_mask = tf.cast(tf.math.not_equal(comparison, previousGuesses), dtype = tf.float32)
            logits = self((concatSequence,  attention_mask))
            assert(logits.shape == (batchSize, concatSequence.shape[1], self.vocab_size))
            
            #get the max of the LAST logits. 
            #try batch size of one. 
            log = logits[:, -1, :]
            if(lookAtPrev):
                log = log*previous_guess_mask
            assert(log.shape == (batchSize, self.vocab_size))
            log = log*maskBos
            log = log*maskPad
            #make it so can't get the end token if less than minimum length. 
            if i<min_length: 
                log = log*maskEos
            if sample: 
                guessNext = self.randomChoiceProbIndex(log, axis=1)
                
            else: 
                guessNext = tf.argmax(log, axis=-1, output_type = tf.int32)

            assert(guessNext.shape == (batchSize,))
            chosenIndices = tf.where(tf.cast(sequenceEnd, tf.bool), guessNext, pad_token_id*tf.ones((batchSize, ), dtype = tf.int32))
            #record the previous guesses so they can't guess the same thing 2 times in a row. 
            if(lookAtPrev):
                previousGuesses = guessNext
            #add newly chosen indices to this new bunch. 
            concatSequence = tf.concat([concatSequence, chosenIndices[..., tf.newaxis]], axis=-1)
            #reevaluate whether each sequence has ended or not. 
            sequenceEnd =tf.cast(tf.math.logical_and(tf.logical_not(chosenIndices == eos_token_id), tf.cast(sequenceEnd, tf.bool)), tf.int32)
            
        return concatSequence
    def randomChoiceProbIndex(self, probs, axis=1):
        """
        Taking in a 2d tensor probs of size batchSize x vocabSize (doesn't necessarily add to one though), it 
        samples randomly an index according to that probability distribution. 
        """
        #renormalize to one
        sum = tf.reduce_sum(probs, axis=-1)
        difference = 1-sum
        addToEach = difference/probs.shape[-1]
        probs = probs + addToEach
        r = tf.expand_dims(tf.random.uniform(shape = (probs.shape[1-axis], )), axis=axis)
        return tf.cast(tf.argmax(tf.math.cumsum(probs, axis=axis)>r, axis=axis), tf.int32)

    def batch_step(self, inputs, training):
        """
        Batch step, done in both training and testing. 
        """
      
        with tf.GradientTape() as tape: 
            logits = self(inputs, training)
            #Use inputs as labels, but will shift it to the right by one and shift the other to the left by one. 
            #add mask to calculation. 
            loss = self.compute_loss(logits, inputs[0], inputs[1])
        if training:
            grad = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        perplexityVal = perplexity(logits, inputs[0], inputs[1])
        self.perplexity.update_state(perplexityVal)
        
        return {"loss": loss, "perplexity": self.perplexity.result()}
    def train_step(self, inputs):
        #accounts for fact that inputs is tuple in tuple for some reason. 
        inputs = inputs[0]
        return self.batch_step(inputs, True)
    def test_step(self, inputs):
        return self.batch_step(inputs, False)
    def resize_token_embeddings(self, size):
        """
        Doesn't really work and allow us to add tokens to the vocabulary. 
        """
        self.transformerBlock.resize_token_embeddings(size)
        print("done")
    def save_weights(self, path):
        if(self.addEmbedding):
            self.head.save_weights(path + "/head")
        if(self.addTail):
            self.tail.save_weights(path + "/tail")
        return 
    def load_weights(self, path):
        if(self.addEmbedding):
            self.head.load_weights(path + "/head")
        if(self.addTail):
            self.tail.load_weights(path + "/tail")
        return 