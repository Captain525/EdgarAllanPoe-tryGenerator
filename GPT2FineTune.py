import tensorflow as tf
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config
class GPT2FineTune(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.transformerBlock = TFGPT2Model.from_pretrained("gpt2")
        self.transformerBlock.resize_token_embeddings(vocab_size)
        self.transformerBlock.trainable = False
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.lossCalc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    def compute_loss(self, logits, labels):
        """
        labels is just the input sequence rn. 
        logits are the probabilities(not normalized) given to each one. 
        want to do sparse categorical crossentorpy after shifting labels by 1 and lgoits by 1.

        assuming labels: (bs, sequenceSize)
        assuming logits: (bs, sequenceSize, vocabSize) 
        """
        labels = labels[:, 1:]
        logits= logits[:,:-1, :]
        #true then predicted. 
        loss = self.lossCalc(labels, logits)
        #print(loss)
        #print(loss.shape)
        #assert(loss.shape == (labels.shape[0],))
        return loss

    def call(self, inputs, training):
        outputs = self.transformerBlock(inputs)
        lastHiddenState = outputs[0]
        logits= self.dense(lastHiddenState)
        return logits

    def batch_step(self, inputs, training):
        with tf.GradientTape() as tape: 
            logits = self(inputs, training)
            #second input should be "labels" not sure if inputs right. 
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