import tensorflow as tf
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config
class GPT2FineTune(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.transformerBlock = TFGPT2Model.from_pretrained("gpt2")
        self.transformerBlock.resize_token_embeddings(vocab_size)
        #want transformerBlock TO NOT TRAIN. TOO MUCH MEMORY REQUIRED, WE WANT TO FINE TUNE IT. 
        self.transformerBlock.trainable = False
        #MLP AT THE END TO PICK SPECIFIC VOCABULARY. 
        self.dense = tf.keras.layers.Dense(vocab_size)
        #loss function here. 
        self.lossCalc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

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

        outputs = self.transformerBlock(inputs)
        #get just the last hidden state of the outputs. 
        lastHiddenState = outputs[0]
        assert(lastHiddenState.shape == (bs, sequenceSize,768))
        #apply dense layer to that hidden state. 
        logits= self.dense(lastHiddenState)
        return logits

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
        self.transformerBlock.resize_token_embeddings(size)
        print("done")