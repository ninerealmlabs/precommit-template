# In this post: Ilya Sutskever's AI Reading list in ~120 words per item

Earlier this year, a reading list with about 30 papers
[was shared on Twitter](https://x.com/andrew_n_carr/status/1752526711311507526).\
It reportedly forms part of a longer version originally compiled by Ilya Sutskever, co-founder and chief scientist of
OpenAI at the time, for John Carmack in 2020 with the remark:

> [_'If you really learn all of these, you'll know 90% of what matters'_.](https://dallasinnovates.com/exclusive-qa-john-carmacks-different-path-to-artificial-general-intelligence/)

While [the list](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE) is fragmentary and much has happened in
the field since, this endorsement and the claim that it was part of onboarding at OpenAI quickly made it go
[somewhat](https://news.ycombinator.com/item?id=40397806)
[viral](https://old.reddit.com/r/ArtificialInteligence/comments/1cpbh1s/ilya_sutskever_if_you_really_learn_all_of_these/).

At about 300,000 words total, the combined content nonetheless corresponds to around one thousand book pages of dense,
technical text and requires a decent investment in time and energy for self-study. After doing just that, I therefore
dedicate this blog post to all those of us who provisionally bookmarked it ("for later") and are still curious. What
follows is my own condensed and structured summary with about 120 words per item, free of mathematical notation, to
capture the essential key points, context and some perspective gained from reading it with the surrounding literature.

## In a Nutshell

The list contains 27 reading items, with papers, blog posts, courses, one dissertation and two book chapters, all
originally dating from 1993 to 2020.

The contents can be roughly broken down as follows:

| Methodology                          | Items | "Percentage of total word count" | Topics                                                                                                                                 |
| ------------------------------------ | ----- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Convolutional Neural Networks (CNNs) | 5     | 25%                              | image recognition, semantic segmentation                                                                                               |
| Recurrent Neural Networks (RNNs)     | 10    | 19%                              | language modeling, speech-to-text, machine translation, combinatorial optimization, visual question answering, content-based attention |
| Transformers                         | 3     | 6%                               | multi-head and dot-product attention, language model scaling                                                                           |
| Information Theory                   | 5     | 42%                              | Kolmogorov complexity, compression, Minimum Description Length                                                                         |
| Miscellaneous                        | 4     | 8%                               | variational inference, representation learning, graph neural networks, distributed training                                            |

Using these categories, the next sections summarize the gist of each item, roughly sorted by how they build on each
other.

## Convolutional Neural Networks

> CS231, 2017 Stanford University Course\
> Length: ~50,000 words, forming 11 blocks of 2 modules\
> Instructors: Fei-Fei Li, Andrej Karpathy and Justin Johnson

[CS231, 2017](https://cs231n.github.io/) is a classic course on deep learning fundamentals from Stanford University. It
builds up from linear classifiers and their ability to learn a given task based on mathematical optimization, or
training, which adjusts their internal parameter weights such that applying them to input data will produce more
desirable outputs. This basic concept is developed into backpropagation for training of neural networks, in which
trainable parameters are typically arranged into multiple layers together with other modules such as activation
functions and pooling layers. _Convolutional Neural Networks_ (CNNs) are introduced as a specialized architecture for
image recognition, as used in modern computer vision systems to this day. Extended video lectures are
[available on youtube](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk).

_Note: If you are starting from zero, this course and newer resources by e.g.
[DeepLearning.AI on Coursera](https://www.coursera.org/specializations/deep-learning) or
[FastAI](https://course.fast.ai/) will help you to get more out of the remaining list._

---

> AlexNet, 2012 Paper\
> Length: ~6,000 words\
> Authors: Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

[AlexNet, 2012](https://dl.acm.org/doi/abs/10.1145/3065386) established CNNs as state of the art for image recognition
and arguably initiated the widespread hype around deep learning. It outperformed its competitors in the 2012 ImageNet
benchmark challenge, predicting whether a given input image contained e.g. a cat, dog, ship or any other of 1,000
possible classes, so conclusively that the real-world dominance of deep learning became commonly accepted. An important
factor was its early\* CUDA implementation that enabled unusually fast training on GPUs.

_\*Note: Earlier GPU implementations are documented in
[section 12.1.2](https://www.deeplearningbook.org/contents/applications.html) of the book
[Deep Learning](https://www.deeplearningbook.org/)._

---

> ResNet, 2015 Paper\
> Length: ~6,000 words\
> Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun

[ResNet, 2015](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
succeeded AlexNet as a more modern CNN architecture, reaching first place on the ImageNet challenge in 2015. It remains
a popular CNN architecture to this day and is
[subject of ongoing research](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html).
It introduced residual connections into CNN architectures that had become ever deeper, stacking more convolutional
layers to achieve higher representational power. By allowing residual connections to skip or bypass entire blocks of
layers, ResNet architectures suffered less from gradient degradation effects in training and could thus be robustly
trained at previously unseen depth.

---

> ResNet identity mappings, 2016 Paper\
> Length: ~6,000 words\
> Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun

[ResNet identity mappings, 2016](https://arxiv.org/abs/1603.05027) were later proposed by the ResNet authors as a
'clean' information path and best design for the skip connections, so that their contents are merely added to the
results of a bypassed block without any further modification. Whereas earlier designs placed an activation layer on the
skip path after the addition, the proposed pre-activation design moves this layer to the start of the bypassed block
instead. The skip connections can thus form a shortcut through the entire neural network that is only interrupted by
additions, allowing improved propagation of gradient signals that make it possible for even deeper neural networks to
be trained.

---

> Dilated convolutions, 2015 Paper\
> Length: ~6,000 words\
> Authors: Fisher Yu and Vladlen Koltun

[Dilated convolutions, 2015](https://arxiv.org/abs/1511.07122) (or _à trous_ convolutions) were proposed as a new type
of module for dense prediction with CNNs in tasks like semantic image segmentation, where class labels are assigned to
any given pixel of an input image. Architectures such as AlexNet and ResNet condense input images to lower-dimensional
representations via strided convolutions or pooling layers to predict one class label for an entire image. Related
architectures for dense prediction therefore typically restore the original input image resolution from these
downsampled, intermediate representations via upsampling operations. Whereas e.g.
[transpose convolutions](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) achieve this with competitive
results, dilated convolutions avoid downsampling entirely. Instead, they space out the filter kernel of a convolutional
layer to skip one or more neighboring input pixels, thereby providing a larger receptive field without any reduction in
resolution.

---

## Recurrent Neural Networks

Today, _Recurrent Neural Networks_ (RNNs) have been largely superseded by Transformers and date from what Ilya
Sutskever himself would later call the
["[pre-2017] stone age"](link:https://www.youtube.com/watch?v=Ft0gTO2K85A&t=625s) of machine learning. They
nonetheless remain [subject of active research](https://arxiv.org/abs/2405.04517) and see continued use in certain
applications. Forming a substantial part of the reading list, they showcase the evolution of early insights and
architectural developments that lead up to the systems of today. Most of the RNNs listed below are
[_Long Short-Term Memory_ (LSTM)](https://link.springer.com/chapter/10.1007/978-3-642-24797-2_4) architectures. Some
designs furthermore include _Feedforward Networks_ with no recurrent connections, usually trained end-to-end as part of
the model.

---

> Understanding LSTM Networks, 2015 Blog Post\
> Length: ~2,000 words\
> Author: Christopher Olah

[Understanding LSTM Networks, 2015](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) provides a brief
introduction to RNNs and LSTMs in particular. RNNs can process a sequence of inputs, one step at a time, while evolving
a hidden state vector that is (re-)ingested, updated and returned again at each step along the input sequence. The
hidden state vector thereby allows for information to persist and be passed to subsequent processing steps.
Nonetheless, simpler RNNs typically struggle with long-term dependencies. LSTMs alleviate this by introducing a cell
state as additional recurrent in- and output, acting as a memory pathway for addition, update or removal of information
along each processing step via trainable gating mechanisms.

---

> The Unreasonable Effectiveness of RNNs, 2015 Blog Post\
> Length: ~6,000 words\
> Author: Andrej Karpathy

[The Unreasonable Effectiveness of RNNs, 2015](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) shows use cases
and results of RNNs in action. They are distinguished by their ability to both process and predict variable-sized
sequences while also maintaining an internal state, prompting the author Andrej Karpathy to state _"If training vanilla
neural nets is optimization over functions, training recurrent nets is optimization over programs."_. He showcases
results for image captioning and character-level language modeling that enables RNNs to automatically generate prose
and articles. Early code generation capabilities are noted, with convincing syntax but failure to compile and a
tendency to suffer from _hallucinations_ where the model provides outputs as most probable that are evidently
incorrect. The blog post also includes
[a minimal RNN code example](https://gist.github.com/karpathy/d4dee566867f8291f086).

---

> RNN Regularization, 2014 Paper\
> Length: ~3,500 words\
> Authors: Wojciech Zaremba, Ilya Sutskever and Oriol Vinyals

[RNN regularization, 2014](https://arxiv.org/abs/1409.2329) addresses the challenge of training large RNNs without
overfitting, where a model would excessively adapt to, or even memorize, its training samples and fail to generalize to
new data. A technique for regularization, which aims to reduce this effect, was proposed that applies _dropout_, which
omits randomly selected outputs of a given neural network layer. Dropout had been known for several years and was used
e.g. in AlexNet. Here, the key insight was to utilize dropout only within a given RNN cell, but to avoid it on the
recurrent connections that carried the hidden state vector. In this way, larger RNNs could avoid overfitting while
preserving long-term dependencies.

---

> Neural Turing Machines, 2014 Paper\
> Length: ~7,500 words\
> Authors: Alex Graves, Greg Wayne and Ivo Danihelka

[Neural Turing Machines, 2014](https://arxiv.org/abs/1410.5401) were proposed as a form of memory-augmented neural
network, with an external memory bank on which an RNN controller could write or erase information with a 'blurry',
differentiable, attention-based focus. Equipped with this working memory, the Neural Turing Machine outperformed a
baseline RNN in experiments involving associative recall, copying and sorting sequences and generalized more robustly
to sequence lengths that exceeded those encountered in training.

---

> Deep Speech 2, 2016 Paper\
> Length: ~7,000 words\
> Authors: Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared
> Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, Jie Chen, Jingdong Chen, Zhijie Chen, Mike Chrzanowski, Adam
> Coates, Greg Diamos, Ke Ding, Niandong Du, Erich Elsen, Jesse Engel, Weiwei Fang, Linxi Fan, Christopher Fougner,
> Liang Gao, Caixia Gong, Awni Hannun, Tony Han, Lappi Johannes, Bing Jiang, Cai Ju, Billy Jun, Patrick LeGresley,
> Libby Lin, Junjie Liu, Yang Liu, Weigao Li, Xiangang Li, Dongpeng Ma, Sharan Narang, Andrew Ng, Sherjil Ozair, Yiping
> Peng, Ryan Prenger, Sheng Qian, Zongfeng Quan, Jonathan Raiman, Vinay Rao, Sanjeev Satheesh, David Seetapun, Shubho
> Sengupta, Kavya Srinet, Anuroop Sriram, Haiyuan Tang, Liliang Tang, Chong Wang, Jidong Wang, Kaifu Wang, Yi Wang,
> Zhijian Wang, Zhiqian Wang, Shuang Wu, Likai Wei, Bo Xiao, Wen Xie, Yan Xie, Dani Yogatama, Bin Yuan, Jun Zhan,
> Zhenyao Zhu

[Deep Speech 2, 2016](http://proceedings.mlr.press/v48/amodei16.html) proposed an automatic
speech recognition system to convert audio recordings into text by processing log-spectrograms representing the audio
with RNNs to predict sequences of characters of either English or Mandarin. The authors utilized batch normalization
instead of dropout for regularization on the non-recurrent layers and _Gated Recurrent Units_ (GRUs) as a somewhat
simplified alternative to LSTMs used in most of the other papers examined so far, together with a plethora of other
engineering tweaks, including batched processing for low-latency streaming output.

---

> RNNsearch, 2015 Paper\
> Length: ~8,000 words\
> Authors: Dzmitry Bahdanau, KyungHyun Cho and Yoshua Bengio

[RNNsearch](https://arxiv.org/abs/1409.0473) is credited with introducing the first attention mechanism into _Natural
Language Processing_ (NLP), proposing additive, _content-based attention_ for neural machine translation. Its
encoder-decoder architecture encodes an input sequence of English words with an RNN encoder into a _context vector_
used by an RNN _de_coder to predict an output sequence of French words. In prior work this context vector was simply
the final hidden state of the encoder, which therefore had to contain all relevant information about the input
sequence. RNNsearch addresses this bottleneck by making the context vector a weighted sum over \_all_ encoder hidden
states, or _annotations_. When predicting a target word, the decoder can thereby rely on context from arbitrary parts
of the encoded input sequence by (re-)calculating the context vector as a weighted sum of annotations. The weighting is
determined by an _alignment model_, a feedforward network that receives the current decoder hidden state together with
an annotation and assigns a score to the latter.

---

> Pointer Networks, 2015 Paper\
> Length: ~4,500 words\
> Authors: Oriol Vinyals, Meire Fortunato and Navdeep Jaitly

[Pointer Networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/29921001f2f04bd3baee84a12e98098f-Paper.pdf)
repurpose the concept of content-based attention to solve combinatorial optimization problems. Here, content-based
attention is used to 'point' at elements of the input sequence in a specific order. The output sequence is therefore an
indexing of the input elements. Given a set of two-dimensional points as input, Pointer Net was trained to solve for
their [convex hull](https://en.wikipedia.org/wiki/Convex_hull),
[Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) or
[Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) by predicting in which order
these points should be visited. With no limitation to the length of the output sequence or dictionary, this approach
was found to generalize beyond the longest sequence length encountered in training.

---

> Set2Set, 2016 Paper\
> Length: ~6,500 words\
> Authors: Oriol Vinyals, Samy Bengio and Manjunath Kudlur

[Set2Set](https://arxiv.org/abs/1511.06391) extends sequence-to-sequences methods as examined above to enable
order-invariant processing of sets. These methods are shown to strongly depend on the specific order of both an input
and output sequence (e.g. the exact order of random points provided to Pointer Networks for convex hull prediction).
The authors propose Set2Set as a solution, with the encoder forming a _memory bank_ (that resembles _annotations_ of
RNNSearch) to create a context vector. This memory bank, however, is sampled more than just once. Instead, a _process
block_ introduces a new LSTM, which evolves a _query vector_ for repeated, content-based attention readouts of the
memory. Finally, the _write block_ (a Pointer Network) can add even more attention steps in the form of _glimpses_.

---

> Relation Networks, 2017 Paper\
> Length: 5,000 words\
> Authors: Adam Santoro, David Raposo, David G. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy
> Lillicrap

[Relation Network, 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html)
modules were proposed as a method for relational inference tasks such as visual and text-based question answering. The
_Relation Network_ module ingests a pair of feature vectors, for example an LSTM hidden state for a word or sentence in
text or the values at a specific pixel position in feature maps produced by a CNN for image data. A given pairing is
processed with one or more neural network layers before forming an element-wise sum and creating an output with a
second stack of layers. By doing this for all pairs of inputs, this approach outperformed the human baseline in
answering textual questions regarding the size, position and color of 3D generated shapes relative to each other.

---

> Relational Recurrent Neural Networks, 2018 Paper\
> Length: 6,000 words\
> Authors: Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol
> Vinyals, Razvan Pascanu, Timothy Lillicrap

[Relational Recurrent Neural Networks, 2018](https://proceedings.neurips.cc/paper/2018/hash/e2eabaf96372e20a9e3d4b5f83723a61-Abstract.html)
proposed a _Relational Memory Core_ module in which an attention mechanism allows memories to interact with each other
and be recurrently refined as a fixed-size matrix. This approach was adapted for several tasks requiring relational
reasoning and outperformed multiple baseline methods. Given a random set of vectors of which an arbitrary one was
marked, it predicted which other vector had the highest Euclidean distance to it. It also learned to execute short code
snippets involving variable manipulation, performed language modeling and scored well in a toy reinforcement learning
task. The _self-attention_ mechanism that enabled its memory interactions is described in the following section.

---

## Transformers

The previous section tracks the rise of attention mechanisms as an increasingly potent tool for providing context in
sequence-to-sequence prediction tasks. Eventually, these developments yielded the Transformer as a neural network
architecture that predominantly relies on attention and discards both recurrent and convolutional layers entirely. The
excellent scalability of this approach, together with growing compute resources and extensive training data,
established Transformers as dominant method for language modeling, forming the backbone of systems like ChatGPT and
performing well even on image and multimodal data.

New attention mechanisms enabled [substantial speed and efficiency advantages](https://ai.stackexchange.com/a/31584):\
The _additive attention_ mechanism of the previous section compared an encoder and decoder hidden state to each other
by applying an alignment model to each such pair for scoring. Internally, the alignment model formed linear projections
of both vectors and added them together to calculate a score, which was normalized over all encoder hidden states to
form a context vector as their weighted average.\
With _multiplicative attention_, Transformers compare multiple pairings of hidden states at once by forming a dot
product of their linear projections, which can be implemented with faster, highly optimized matrix multiplications.

---

> Attention Is All You Need, 2017 Paper\
> Length: ~4,500 words\
> Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and
> Illia Polosukhin

[Attention Is All You Need](https://user.phil.hhu.de/~cwurm/wp-content/uploads/2020/01/7181-attention-is-all-you-need.pdf)
proposed the Transformer architecture. In an encoder-decoder structure for machine translation, embedding layers
convert each input token into a feature vector, to which positional encodings are added. The proposed _Scaled
Dot-Product Attention_ computes a weighted average over multiple _value_ vectors, each weighted by comparing its
associated _key_ vector to a given _query_ vector using the dot product. The result is scaled (for numerical stability)
and then normalized over all keys with a softmax function. _Multi-head attention_ conducts this process in parallel
with different, learned projections of each input.\
Three variants of this mechanism are used. In _self-attention_ used by the encoder, the _query_, _key_ and _value_ are
distinct linear projections of the same output vector from the previous layer. In _masked self-attention_ the decoder
furthermore masks out the weights for future tokens. Finally, in _encoder-decoder attention_, each decoder block
obtains only the _query_ from the preceding decoder layer, whereas _key_ and _value_ originate from the final encoder
layer. The experiments exceeded state-of-the-art results, with two orders of magnitude lower compute resources than
previous approaches.

---

> The Annotated Transformer, 2020 Blog Post (2022 version)\
> Length: ~6,000 words\
> Authors: Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak, and Stella Biderman (2020 original by Sasha
> Rush)

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) implements the Transformer as
described in _'Attention is All You Need'_ line by line as a fully functional Jupyter Notebook using PyTorch, with all
[code available on GitHub](https://github.com/harvardnlp/annotated-transformer/). Text segments of the original paper
feature alongside the code, together with comments and visualizations that clarify various aspects of the architecture
beyond the contents of the paper. The notebook also implements examples for data formatting, training and inference
that show the Transformer applied in practice.

_Note: [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) is yet another
in-depth guide._

---

> Scaling Laws for Neural Language Models, 2020 Paper\
> Length: ~9,000 words\
> Authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec
> Radford, Jeffrey Wu and Dario Amodei

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) explores the predictive performance of
Transformers for language modeling as a function of model size, data quantity and available compute resources.
Extensive empirical results enable the authors to establish formulas that relate these factors to each other over seven
orders of magnitude and enable several recommendations as to their optimal configuration. While each of these can form
a bottleneck, model size (i.e. the number of trainable parameters) forms the single most impactful factor. Larger
models reach higher sample-efficiency and better generalization earlier on in training. The specific model architecture
has little effect. An eight-fold increase in model size requires a five-fold increase in training data. Given a fixed
compute budget, the authors accordingly recommend to prioritize first model size, then batch size and only then the
number of training steps, with early stopping before convergence typically providing the best trade-off in their
experiments.

---

## Information Theory

A substantial portion of the reading list is dedicated to more abstract material on theoretical informatics. Rather
than proposing specific architectures or engineering solutions for concrete applications, these works are concerned
with more fundamental study of the limits of computability, probability and intelligence. Recurring themes are
principles for inductive inference such as [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor), which states
a preference for simplicity when choosing between competing explanations (be they theories, hypotheses or models) for
some given evidence or data. Another core concept is
[Kolmogorov complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity)\* for quantifying the amount of
information, or potential for compression, of a given input.

_\*Note: Kolmogorov complexity of a sequence can be defined as the length of the shortest program that prints it and
then halts. While uncomputable in practice, it can be approximated with compression software such as gzip._

---

> A Tutorial Introduction to the Minimum Description Length Principle, 2004 Book Chapter\
> Length: ~30,000 words\
> Author: Peter Grünwald

[A Tutorial Introduction to the Minimum Description Length Principle, 2004](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf)
describes an approach for model selection that mathematically formalizes Occam's razor, defining a preference for the
most simple model among all those that explain the available data. The principle relates learning to data compression,
as the ability to exploit regularity for achieving a shortest possible description. This description is defined by
codes, and codelength functions are noted as corresponding to probability mass functions. The _two-part code version_
of the Minimum Description Length (MDL) principle measures the simplicity of a model instance as the length of its
description (in bits) added to the length of the data description as encoded with it. The _refined, one-part code
version_ examines entire families of models based on their goodness-of-fit and complexity.

---

> Kolmogorov Complexity and Algorithmic Randomness\
> (Chapter 14), 2017 Book Chapter\
> Length: ~35,000 words\
> Authors: Alexander Shen, Vladimir A. Uspensky and Nikolay Vereshchagin

[Kolmogorov Complexity and Algorithmic Randomness](https://www.lirmm.fr/~ashen/kolmbook-eng-scan.pdf) features a final
chapter on algorithmic statistics. In this framework, a given sequence of observations is encoded as one binary string.
Kolmogorov complexity provides formal means of quantifying its randomness and regularity, as well as the expected and
desired properties of a theory or model that can explain it. Such a model should preferably be simple, as indicated by
low Kolmogorov complexity. It should also explain as much regularity in the data as possible, making the data "typical"
for the model. This property is formally quantified by low _randomness deficiency_ of the data relative to the model.
Together, these two properties are also related to the two-part code of the Minimum Description Length principle. The
chapter closes by drawing parallels between good models and good compressors, together with the potential of lossy
compression to perform effective denoising.

---

> The First Law of Complexodynamics, 2011 Blog Post\
> Length: ~2,000 words\
> Author: Scott Aaronson

[The First Law of Complexodynamics](https://scottaaronson.blog/?p=762) explores the relationship between entropy and
complexity. Whereas the second law of thermodynamics dictates that entropy of closed systems increases over time, their
complexity of 'interestingness' is noted to first rise and then fall again. Giving the example of coffee and milk
mixing in a glass, the highest such 'complextropy' is noted to occur midway, when tendrils of milk result from both
liquids no longer being cleanly separated but also not yet forming a homogeneous blend. Kolmogorov complexity is
explored as a way to express both entropy and this 'complextropy', with the conjecture that a resource-bounded
definition could provide a suitable theoretical framework.

---

> Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton, 2014 Paper\
> Length: ~8,500 words\
> Authors: Scott Aaronson, Sean M. Carroll and Lauren Ouellette

[Quantifying the Rise and Fall of Complexity](https://arxiv.org/abs/1405.6903) explores these ideas in further depth.
Covering various theoretical notions of complexity, it eventually settles on 'apparent complexity' as a way of modeling
the separate phenomena of entropy and the 'interestingness' of a closed system. Practical experiments inspired by the
blending of coffee and milk fill a 2D array with a clean split of binary values and perturb these over multiple time
steps to represent random mixing. This array forms an image which is compressed by gzip to approximate Kolmogorov
complexity via file size. At each time step, this is done with the image itself to approximate entropy, but also with a
coarse-grained, blurred version to estimate its apparent complexity. As envisioned, the increasingly noisy image values
yield rising entropy whereas their blurred representation first raises and then decreases the apparent complexity
measure as the mix gets more homogeneous.

---

> Machine Super Intelligence, 2008 Dissertation\
> Length: ~50,000 words\
> Author: Shane Legg, supervised by Marcus Hutter

[Machine Super Intelligence, 2008](https://sonar.ch/usi/documents/317954) explores universal artificial intelligence
under aspects of algorithmic complexity, probability and information theory. It covers inductive inference from
Epicurus principle of multiple explanations, Occam's razor, Bayes rule and priors to complexity measures and
agent-environment models as examined in reinforcement learning. Discussing various definitions and established tests
for intelligence, it proposes a formal definition and measure for universal intelligence\* as the ability of an agent
to achieve specific goals in a wide range of environments. While the proposed measure itself is uncomputable in
practice, it enables theoretical conclusions, such as the requirement that powerful agents be proportionally complex,
and motivates several practical experiments in which a downscaled version of a hypothetically optimal agent is deployed
for reinforcement learning.

\*_Note: This measure would accordingly score the universal intelligence of specialized, 'narrow' machine learning
systems that form the bulk of the papers examined in this blog post as comparatively low._

---

## Miscellaneous

---

> Keeping Neural Networks Simple by Minimizing the Description Length of the Weights, 1993 Paper\
> Length: ~6,000 words\
> Authors: Geoffrey E. Hinton and Drew van Camp

[Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)
introduced the concept of Variational Inference with neural networks. This approach enables neural network training to
approximate the otherwise computationally prohibitive concept of Bayesian inference. The authors propose a
regularization technique that represents each weight of a neural network as a Gaussian probability distribution
described by a mean and a variance value. Inspired by the Minimum Description Length principle, the cost function used
during training penalizes the description length of the weights and the data misfits. The authors argue that this
representation allows for a substantial reduction in the description length of the weights. Their _Bits-Back Coding_
argument states that the distribution of each weight can be sampled with random bits at no additional cost, as the
random bits can be reconstructed given a fixed learning algorithm, architecture and initial probability distribution
for each weight.

---

> Variational Lossy Autencoder, 2017 Paper\
> Length: ~6,000 words\
> Authors: Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever and
> Pieter Abbeel

[Variational Lossy Autoencoders](https://arxiv.org/abs/1611.02731) provide a way for data compression with control over
which aspects of the data should be retained or discarded. In experimental results, this enables 2D image compression
that discards local texture while retaining global structure. Autencoders use an inference model to compresses input
data to a compact _latent code_, from which a generative model decodes the original input. This latent code should
accordingly represent all information relevant for describing the input. When using sufficiently powerful
autoregressive models like RNNs however, decoders had been previously found capable of predicting the output while
ignoring the latent code entirely. Here, a theoretical explanation for this phenomenon is provided based on Bits Back
Coding. The proposed approach weakens the decoder (e.g. limiting it to reconstruct small receptive fields) such that it
depends on the missing information (e.g. global structure) being fully provided by the latent code to which the input
is compressed.

---

> GPipe, 2018 Paper\
> Length: ~5,000 words\
> Authors: Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan
> Ngiam, Quoc V. Le, Yonghui Wu and Zhifeng Chen

[GPipe](https://arxiv.org/abs/1811.06965) is\* a library for distributed training of neural networks on more than one
accelerator (e.g. GPUs). It subdivides the neural network architecture into _cells_ formed by one or more consecutive
layers and assigns each cell to a separate accelerator. It furthermore employs pipeline parallelism by also splitting
each mini-batch of training samples into several _micro_-batches that are pipelined through, so that multiple
accelerators can work on different micro batches concurrently. The gradients for all micro-batches are aggregated for
one synchronous update per mini-batch. Training thus remains consistent regardless of cell count or micro-batch size.

\*_Note: As for the library itself, [GitHub](https://github.com/kakaobrain/torchgpipe) shows that its most recent
commit for the final version v0.0.7 occurred in September 2020._

---

> Neural Message Passing for Quantum Chemistry, 2017 Paper\
> Length: ~6,000 words\
> Authors: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals and George E. Dahl

[Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) explores the application of graph
neural networks to predict quantum mechanical properties of organic molecules. The commonalities between several
related works that utilize neural networks for graph data are first discussed and abstracted into a new concept of
_Message Passing Neural Networks_. This framework considers undirected graphs composed of edges and nodes, both of
which can have features. Each forward pass performs one or more steps of a message passing phase in which the hidden
state of each given node is updated based on messages that depend on the hidden states and connecting edges with all
adjacent nodes. Next, a readout phase calculates one hidden state for the entire graph using a readout function that is
invariant to the order of graph nodes. Using the [Gated Graph Neural Network](https://arxiv.org/abs/1511.05493)
architecture, the authors present experimental results on graph data of molecular structures that achieved
state-of-the-art results at the time.

---

## Concluding Thoughts

Whereas this summary was written by hand, the described technology is approaching a point where its language modeling
capabilities are hard to distinguish from human writing already now in 2024. Eventually, Large Language Models may
indeed become self-explanatory in a literal sense. Prompting e.g. ChatGPT to summarize this material nonetheless still
yields explanations that _seem_ very convincing but can also be largely hallucinated and often misleading. Perhaps that
will already improve once this article is ingested into the training data?

Although low-quality, generated content was a recurring theme I encountered while researching this list, there are also
several independent summaries of the reading list worth sharing here:

- [Aman Chadha's _Distilled AI_](https://aman.ai/primers/ai/top-30-papers/) (~12,000 words, includes other papers)
- [DataMListic's youtube video playlist](https://www.youtube.com/watch?v=oF6vcYHs6rw&list=PL8hTotro6aVGtPgLJ_TMKe8C8MDhHBZ4W)
  (about 25 min total)

With this blog post, the known contents of the reading list are compressed to barely more than one percent of the
original word count. This leaves a lot more to be discussed, but hopefully it still has something to offer for the
interested reader. My own, subjective review will be saved for another post in the future.
