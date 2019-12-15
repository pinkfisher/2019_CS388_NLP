# 2019 CS388 NLP Projects

## Mini 1: Classification for Person Name Detection

In this project you’ll implement a simple classifier that can determine whether a token is part of a person’s name. 
This will reinforce the basics of classification (which you should have seen before) and teach you the basics of large-scale machine learning with sparse feature vectors, including techniques for feature extraction, feature indexing, optimization, etc.

## Project 1: Sequential CRF for NER

In this project you’ll implement a CRF sequence tagger for NER. You’ll implement the Viterbi algorithm on a fixed model first (an HMM), then generalize that to forward-backward and implement learn- ing and decoding for a feature-based CRF as well. The primary goal of this assignment is to expose you to inference and learning for a simple structured model where exact inference is possible. Secondarily, you will learn some of the engineering factors that need to be considered when implementing a model like this.

## Mini 2: Neural Networks for Sentiment Analysis

In this project, you will implement two different neural networks for sentiment analysis: a feedfor- ward “deep averaging” network in the style of Iyyer et al. (2015) and either an RNN or CNN-based approach of your choosing. The goal of this project is to give you experience implementing standard neural network architectures in Pytorch for an NLP task.

## Project 2: Semantic Parsing with Encoder-Decoder Models

 In this project you’ll implement an encoder-decoder model for semantic parsing. A sample encoder implementation is provided to you and conceptually resembles the encoder you built in Mini 2. You will have to figure out how to implement the decoder module, combine it with the encoder, do training, and do inference. Additionally, you’ll be exploring attention as well as some other features of encoder-decoder models as part of your extension.
