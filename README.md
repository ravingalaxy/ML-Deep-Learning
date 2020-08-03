<div align="center">
    <img src="./images/Udacity.png">
    <h1>Udacity Deep Learning Nanodegree</h1>
    <p>Become an expert in neural networks, and learn to implement them using the deep learning framework PyTorch. Build convolutional networks for image recognition, recurrent networks for sequence generation, generative adversarial networks for image generation, and learn how to deploy models accessible from a website. - <a href="https://www.udacity.com/course/deep-learning-nanodegree--nd101">Source</a></p>
    <p>The Deep Learning Nanodegree program is divided into five parts, giving a understanding of deep learning, and covering some of the major topics.
</p>
</div>
<br>    
<hr style="height:3px">
<br>

<!--Part First-->
<h2>Introduction</h2>

<p>The first part is an introduction to the program as well as a couple lessons covering
tools. Here I also got a chance to apply some deep learning models to do
cool things like transferring the style of artwork to another image.</p>

<p>Here I started with a simple introduction to linear regression and machine learning. That
gave me the vocabulary I need to understand the advancements, and made
clear where deep learning fits into the broader picture of machine learning techniques.</p>
<br>
<hr style="height:3px">
<br>
<!--Part Second-->

<h2>Neural Networks</h2>
<p>In this part, I learnt how to build a simple neural network from scratch using
python. I covered the algorithms used to train networks such as gradient descent and
backpropagation.</p>

<p>The <b>first project</b> (<a href="https://github.com/ravingalaxy/ML-Deep-Learning/tree/master/Projects/Project_1-Predicting_Bike_Sharing_Patterns">PREDICTING BIKE-SHARING PATTERNS</a>) is also created in this part. In this project, I predicted bike ridership
using a simple neural network.</p>
<br>
<div align="center"><img src="./images/neural.PNG">
<br>
<small>Multi-layer neural network with some inputs and a single output. Image from <a href="https://cs231n.github.io/convolutional-networks/">Stanford's cs231n course.</a>
</small>
</div>

<h3>Sentiment Analysis</h3>

<p>I also learnt about model evaluation and validation, an important technique for
training and assessing neural networks. And I got a chance to learn Sentiment Analysis from a guest instructor Andrew Trask,
author of <a href="https://www.manning.com/books/grokking-deep-learning">Grokking Deep Learning</a>, developing a neural network for processing text and predicting sentiment. The exercises in each chapter of his book are also available in
his <a href="https://github.com/iamtrask/Grokking-Deep-Learning">Github repository</a>.</p>

<h3>Deep Learning with PyTorch</h3>
<p>The last lesson in this part is all about using the deep learning framework, PyTorch. Here,
I learnt how to use the Tensor datatype, and the foundational knowledge I
need to define and train my own deep learning models in PyTorch!</p>
<br>
<div align="center"><img src="./images/pytorch.PNG">
<br>
</div>
<br>
<hr style="height:3px">
<br>    
    
<!--Part Third-->

<h2>Convolutional Networks</h2> 
    
<p>Convolutional networks have achieved state of the art results in computer vision. These
types of networks can detect and identify objects in images. Here I learnt how to build
convolutional networks in PyTorch.</p>
<p>Here I also created the <b>second project</b> (<a href="#">DOG-BREED CLASSIFIER</a>), where i built a convolutional network to classify dog breeds in pictures.</p>
<br>

<div align="center"><img src="./images/cnn.PNG">
<br>
<small>Structure of a convolutional neural network.</small>
</div>

<p>I also used convolutional networks to build an <i>autoencoder</i>, a network architecture
used for image compression and denoising. Then, I use a pre-trained neural
network, to classify images the network has never seen before, a technique known
as <i>transfer learning</i>.</p>
<br>
<hr style="height:3px">
<br>    
    
<!--Part Forth-->

<h2>Recurrent Neural Networks</h2> 

<p>In this part, I learnt about <b>Recurrent Neural Networks (RNNs)</b> — a type of network
architecture particularly well suited to data that forms sequences like text, music, and
time series data. I built a recurrent neural network that can generate new text
character by character.</p>
<br>

<div align="center"><img src="./images/rnn.PNG">
<br>
<small>Examples of input/output sequence types.</small>
</div>

<h3>Natural Language Processing</h3>

<p>Then, I learnt about word embeddings and implement the <a href="https://en.wikipedia.org/wiki/Word2vec">Word2Vec</a> model, a network that can learn about semantic relationships between words. These are used to increase the efficiency of networks when I am processing text.</p>
<p>I combined embeddings and an RNN to predict the sentiment of movie reviews, an
example of common tasks in natural language processing.</p>
<p>In the <b>third project</b> (<a href="#">GENERATE TV SCRIPTS</a>), I used what I had learnt here to generate new TV scripts
from provided, existing scripts.</p>
<br>

<div align="center"><img src="./images/nlp.PNG">
<br>
<small>An example RNN structure in which an encoder represents the question: "how are you?" and a
decoder generates the answer: "I am good".</small>
</div>
<br>
<hr style="height:3px">
<br>

<!--Part Fifth-->

<h2>Generative Adversarial Networks</h2>

<p>Generative adversarial networks (GANs) are one of the newest and most exciting deep learning architectures, showing incredible capacity for understanding real-world data.</p>

<p>In this part, I learnt about and implement GANs for a variety of tasks. You'll even see how to code a <a href="https://github.com/junyanz/CycleGAN">CycleGAN</a> for generating images, and learnt from one of the creators of this formulation, Jun-Yan Zhu, a researcher at <a href="https://www.csail.mit.edu/">MIT's CSAIL</a>.</p>
<br>

<div align="center"><img src="./images/i2i.PNG">
<br>
<small>Examples of image-to-image translation done by <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">CycleGAN and Pix2Pix</a> formulations.</small>
</div>
<br>

<p>
The inventor of GANs, Ian Goodfellow, will show me how GANs work and how to
implement them. Then, in the <b>fourth project</b> (<a href="#">GENERATE FACES</a>), I used a deep convolutional GAN to
generate completely new images of human faces.
</p>
<br>

<div align="center"><img src="./images/fcg.PNG">
<br>
<small>Low-res, GAN-generated images of faces.</small>
</div>
<br>
<hr style="height:3px">
<br>

<!-- Part Six -->
<h2>Deploying Machine Learning Models</h2>

<p>As more and more companies look to build AI products, there is a growing demand for
engineers who are able to deploy machine learning models to global audiences. In this
part, I got experience deploying a model so that it can be accessed via a web app
and respond to user input.</p>
<br>

<div align="center"><img src="./images/cs.PNG">
<br>
<small>Image of a cloud service connecting data to a laptop.</small>
</div>
<br>

<p>You'lI also learnt to monitor a model using PyTorch and <a href="https://aws.amazon.com/sagemaker/">Amazon's SageMaker</a>. In the <b>fifth project</b> (<a href="#">DEPLOYING A SENTIMENT ANALYSIS MODEL</a>), I deployed my own PyTorch sentiment analysis model and create a gateway for accessing this model from a website.</p>
<br>
<hr style="height:3px">
<br>

<!--Projects list-->

<h2>Projects I Have Built</h2>

<p>The five projects I have completed in this course are perhaps the most important part of my learning journey. They gave me the chance to apply what i had learnt and even share my work with friends or potential employers! These projects are designed to be challenging and interesting. I got feedback from a real person; a Udacity reviewer. This reviewer helped me figure out what could be improved in my code and i submited my code again until I pass the project.
</p>
<p>Here are the five projects, created by me:</p>
<br>
1. <a href="https://github.com/ravingalaxy/ML-Deep-Learning/tree/master/Projects/Project_1-Predicting_Bike_Sharing_Patterns">PREDICTING BIKE-SHARING PATTERNS</a>
<br>
2. <a href="#">DOG-BREED CLASSIFIER</a>
<br>
3. <a href="#">GENERATE TV SCRIPTS</a>
<br>
4. <a href="#">GENERATE FACES</a>
<br>
5. <a href="#">DEPLOYING A SENTIMENT ANALYSIS MODEL</a>
<br>
