# EdgarAllanPoe-tryGenerator
Uses a generative neural network to generate Poetry adhering to a specific rhyme scheme. 


Using the pretrained model gpt2, which is a transformer. 

Two ways to customize pretrained model:
1. Feature extraction - Use representations learned from previous model to get meaningful features for new model. Just add a classification network on the end of the pretrained model, train that from scratch, WITHOUT training the pretrained model at all. 

2. Fine tuning - in addition to adding new layers, unfreeze the few top layers (ie the latest laters) and update their weights as well. Thus, you fine tune te model for the specific task. However, leave the earlier layers as is. 

Why fine tune?: Later layers in the network more specific to the task at hand, whereas earlier layers are more general about language and words. thus, we want to tune the model to our specific task. 