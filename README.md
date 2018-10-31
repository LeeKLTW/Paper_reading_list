# Paper_reading_list
A list to record the papers I am reading.


## RNN (Structure)

|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
Sequence to sequence learning with neural networks|Sutskever et al |2014| [pdf](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)||| [Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Sutskever%20et%20al%2CSequence%20to%20sequence%20learning%20with%20neural%20networks.md)|
Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation|Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio|_3 Sep 2014_ | [arxiv](https://arxiv.org/pdf/1406.1078.pdf)||| 
Get To The Point: Summarization with Pointer-Generator Networks|Abigail See, Peter J. Liu, Christopher D. Manning|_25 Apr 2017_ | [arxiv](https://arxiv.org/pdf/1704.04368.pdf)||| 
Recent Advances in Recurrent Neural Networks|Hojjat Salehinejad, Sharan Sankar, Joseph Barfett, Errol Colak, Shahrokh Valaee|_22 Feb 2018_ | [arxiv](https://arxiv.org/pdf/1801.01078.pdf)|||[TBC](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Recent%20Advances%20in%20Recurrent%20Neural%20Networks.md)|
Semi-Supervised Sequence Modeling with Cross-View Training|Tsung-Hsien Wen, Minh-Thang Luong |_19 Sep 2018_ | [arxiv](https://arxiv.org/pdf/1809.07070v1.pdf)||||

## Text(NLP,NLU)
### syntactic parsing
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
|A Fast and Accurate Dependency Parser using Neural Networks|Danqi Chen,Christopher D. Manning|_2014_|[standford](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf)|[not officail github](https://github.com/akjindal53244/dependency_parsing_tf)|EMNLP,Citation(914)||

[哈工大](https://ltp-cloud.com/intro/)

### word embedding
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
| Bag of Tricks for Efficient Text Classification || _April 3-7, 2017_ | [pdf](http://aclweb.org/anthology/E17-2068) | [facebookresearch/fastText](https://github.com/facebookresearch/fastText) | _None_ | [Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.md)|
| Enriching Word Vectors with Subword Information || _July 15, 2016_ | [arxiv](https://arxiv.org/pdf/1607.04606v1.pdf) | [facebookresearch/fastText](https://github.com/facebookresearch/fastText) | _None_ |[Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Enriching%20Word%20Vectors%20with%20Subword%20Information.md)|
Deep contextualized word representations|Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer|_22 Mar 2018_| [arxiv](https://arxiv.org/pdf/1802.05365.pdf)||Citations (137)||
Learned in Translation: Contextualized Word Vectors|Bryan McCann, James Bradbury, Caiming Xiong, Richard Socher|_20 Jun 2018_| [arxiv](https://arxiv.org/pdf/1708.00107.pdf)||Citations (74)||
Analogical Reasoning on Chinese Morphological and Semantic Relations|Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du|_12 May 2018_| [arxiv](https://arxiv.org/pdf/1805.06504.pdf)|[github](https://github.com/Embedding/Chinese-Word-Vectors)|ACL Short Papers,Citations (2)|[read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Analogical%20Reasoning%20on%20Chinese%20Morphological%20and%20Semantic%20Relations.md)|
Semi-Supervised Sequence Modeling with Cross-View Training|Kevin Clark, Minh-Thang Luong, Christopher D. Manning, Quoc V. Le|_22 Sep 2018_| [arxiv](https://arxiv.org/pdf/1809.08370v1.pdf)||||
Devlin et al.BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ||11 Oct 2018| [arxiv](https://arxiv.org/pdf/1810.04805.pdf)||_maybe_beyond_state_of_art_at_LM_||

[Comparison of FastText and Word2Vec in nbviewer.jupyter.org](http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb)

### Translation
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
| Neural Machine Translation by Jointly Learning to Align and Translate|Dzmitry Bahdanau, KyungHyun Cho,Yoshua Bengio. | _19 May 2016_ | [arxiv](https://arxiv.org/pdf/1409.0473.pdf) |  | _None_ | |
| Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation|Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean.  | _8 Oct 2016_ | [arxiv](https://arxiv.org/pdf/1609.08144.pdf) |  | _None_ | |
| The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation|Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Mike Schuster, Zhifeng Chen, Yonghui Wu, Macduff Hughes. | _27 Apr 2018_ | [arxiv](https://arxiv.org/pdf/1804.09849.pdf) |  | _None_ | |


Thang Luong,[NMT](https://github.com/lmthang/thesis/blob/master/thesis.pdf)|[tf-github](https://github.com/tensorflow/nmt)

### Chatbots
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
| A Neural Conversational Model |Oriol Vinyals, Quoc Le|_19 jun 2015_ | [arxiv](https://arxiv.org/pdf/1506.05869) | [inikdom/neural-chatbot](https://github.com/inikdom/neural-chatbot) | _None_ |[Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/A%20Neural%20Conversational%20Model.md)|
| Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models |Iulian V. Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, Joelle Pineau| _6 Apr 2016_ | [arxiv](https://arxiv.org/pdf/1507.04808.pdf) |  | _None_ | |
| A Deep Reinforcement Learning Chatbot |Iulian V. Serban, Chinnadhurai Sankar, Mathieu Germain, Saizheng Zhang, Zhouhan Lin, Sandeep Subramanian, Taesup Kim, Michael Pieper, Sarath Chandar, Nan Rosemary Ke, Sai Rajeshwar, Alexandre de Brebisson, Jose M. R. Sotelo, Dendi Suhubdy, Vincent Michalski, Alexandre Nguyen, Joelle Pineau, Yoshua Bengio| _7 sep 2017_ | [arxiv](https://arxiv.org/pdf/1709.02349) |  | _None_ | |

### Taskbots
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
| Composite Task-Completion Dialogue Policy Learning via Hierarchical Deep Reinforcement Learning|Baolin Peng, Xiujun Li, Lihong Li, Jianfeng Gao, Asli Celikyilmaz, Sungjin Lee, Kam-Fai Wong| _10 apr 2017_ | [arxiv](https://arxiv.org/pdf/1704.03084) | [MiuLab/TC-Bot](https://github.com/MiuLab/TC-Bot) | _None_ ||
| End-to-End Task-Completion Neural Dialogue Systems|Xiujun Li, Yun-Nung Chen, Lihong Li, Jianfeng Gao, Asli Celikyilmaz| _3 mar 2017_ | [arxiv](https://arxiv.org/pdf/1703.01008) | [MiuLab/TC-Bot](https://github.com/MiuLab/TC-Bot) | _None_ | |

### Question Answering
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
|Attention-Based Convolutional Neural Network for Machine Comprehension|Wenpeng Yin, Sebastian Ebert, Hinrich Schütze| _13 Feb 2016_ | [arxiv](https://arxiv.org/pdf/1602.04341v1.pdf) | |MCTest,Citations (42) | |
|Joint Learning of Sentence Embeddings for Relevance and Entailment|Petr Baudis, Silvestr Stanko, Jan Sedivy| _22 Jun 2016_ | [arxiv](https://arxiv.org/pdf/1605.04655v2.pdf) | |MCTest,Citations (2) | |
|A Parallel-Hierarchical Model for Machine Comprehension on Sparse Data|Adam Trischler, Zheng Ye, Xingdi Yuan, Jing He, Phillip Bachman, Kaheer Suleman| _29 Mar 2016_ | [arxiv](https://arxiv.org/pdf/1603.08884v1.pdf) |[github](https://github.com/Maluuba/mctest-model)|MCTest,Citations (19) | |
|Machine Comprehension Using Match-LSTM and Answer Pointer|Shuohang Wang, Jing Jiang| _Nov 2016_ | [arxiv](https://arxiv.org/pdf/1608.07905.pdf) | |SQuAD,Citations (182) | |
| IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models || _30 may 2017_ | [arxiv](https://arxiv.org/pdf/1705.10513) | [geek-ai/irgan](https://github.com/geek-ai/irgan) | ![state-of-the-art](https://img.shields.io/badge/label-State--of--the--art-red.svg) | |
| Making Neural QA as Simple as Possible but not Simpler |Dirk Weissenborn, Georg Wiese, Laura Seiffe| _8 June 2017_ | [arxiv](https://arxiv.org/abs/1703.04816) |  |Citations (40)| |
|Supervised and Unsupervised Transfer Learning for Question Answering|Yu-An Chung, Hung-Yi Lee, James Glass| _21 Apr 2018_ | [arxiv](https://arxiv.org/pdf/1602.04341v1.pdf) | |MCTest,Citations (7) | |
| FQuAC : Question Answering in Context |Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, Luke Zettlemoyer| _28 Aug 2018_ | [arxiv](https://arxiv.org/pdf/1808.07036v3.pdf)| |Citations (6)| |
| Finding Similar Medical Questions from Question Answering Websites |Yaliang Li, Liuyi Yao, Nan Du, Jing Gao, Qi Li, Chuishi Meng, Chenwei Zhang, Wei Fan| _14 Oct 2018_ | [arxiv](https://arxiv.org/df/1810.05983v1.pdf)| |Citations (0)| |

[MCtest](https://mattr1.github.io/mctest/results.html)
![mctest](https://github.com/LeeKLTW/Paper_reading_list/blob/master/MCTEST.jpg)

## Summarization
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
|Automatic Keyword Extraction for Text Summarization: A Survey|Santosh Kumar Bharti, Korra Sathya Babu|_11 Apr 2017_ |[arxiv](https://arxiv.org/pdf/1704.03242v1.pdf)||Citations (6)||
|Text Summarization Techniques: A Brief Survey|Mehdi Allahyari, Seyedamin Pouriyeh, Mehdi Assefi, Saeid Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, Krys Kochut|_28 Jul 2017_ |[arxiv](https://arxiv.org/pdf/1707.02268v3.pdf)||Citations (11)||
|A Brief Survey of Text Mining: Classification, Clustering and Extraction Techniques|Mehdi Allahyari, Seyedamin Pouriyeh, Mehdi Assefi, Saied Safaei, Elizabeth D. Trippe, Juan B. Gutierrez, Krys Kochut|_28 Jul 2017_ |[arxiv](https://arxiv.org/pdf/1707.02919v2.pdf)||Citations (27)||
|A Comprehensive Survey of Ontology Summarization: Measures and Methods|Seyedamin Pouriyeh, Mehdi Allahyari, Krys Kochut, Hamid Reza Arabnia|_5 Jan 2018_|[arxiv](https://arxiv.org/pdf/1801.01937)||Citations (0)||
|A Survey on Neural Network-Based Summarization Methods|Yue Dong|_19 Mar 2018_ |[arxiv](https://arxiv.org/pdf/1804.04589.pdf)||||
|Graph-based Ontology Summarization: A Survey|Seyedamin Pouriyeh, Mehdi Allahyari, Qingxia Liu, Gong Cheng, Hamid Reza Arabnia, Yuzhong Qu, Krys Kochut|_15 May 2018_|[arxiv](https://arxiv.org/pdf/1805.06051.pdf)||Citations (0)||

## Ontology
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
|Deriving a Representative Vector for Ontology Classes with Instance Word Vector Embeddings|Vindula Jayawardana, Dimuthu Lakmal, Nisansa de Silva, Amal Shehan Perera, Keet Sugathadasa, Buddhi Ayesha|_8 Jun 2017_|[arxiv](https://arxiv.org/pdf/1706.02909.pdf)|
|Structural-fitting Word Vectors to Linguistic Ontology for Semantic Relatedness Measurement |Yang-Yin Lee,Ting-Yu Yen,Hen-Hsen Huang,Hsin-Hsi Chen|_Singapore, Singapore — November 06 - 10, 2017_|[pdf](http://nlg.csie.ntu.edu.tw/~hhhuang/docs/cikm2017.pdf)||2017 ACM International Conference on Information and Knowledge Management||
|Domain Ontology Induction using Word Embeddings|Niharika Gupta,Sanjay Podder, Annervaz K M, Shubhashis Sengupta|_20 Apr 2018_|[pdf](https://www.computer.org/csdl/proceedings/icmla/2016/6167/00/07838131.pdf)||2016 15th IEEE International Conference on Machine Learning and Applications||
|Inseparability and Conservative Extensions of Description Logic Ontologies: A Survey|Elena Botoeva, Boris Konev, Carsten Lutz, Vladislav Ryzhikov, Frank Wolter, Michael Zakharyaschev|_20 Apr 2018_|[arxiv](https://arxiv.org/pdf/1804.07805v1)||Citations (10)||


## General and other(activation function, nodes type etc.)
|Title|Author|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|---|
|Pointer Networks|Oriol Vinyals, Meire Fortunato, Navdeep Jaitly|_2 Jan 2017_ |[arxiv](https://arxiv.org/pdf/1506.03134.pdf)||||
Klambauer et al,Self-Normalizing Neural Networks ||8 Jun 2017 | [arxiv](https://arxiv.org/pdf/1706.02515.pdf)|https://github.com/bioinf-jku/SNNs || 
A Brief Survey of Deep Reinforcement Learning |Kai Arulkumaran, Marc Peter Deisenroth, Miles Brundage, Anil Anthony Bharath|8 _28 Sep 2017_ | [arxiv](https://arxiv.org/pdf/1708.05866v2.pdf)| |IEEE Signal Processing Magazine, Special Issue on Deep Learning for Image Understanding (arXiv ext||




# github
[zake7749/Chatbot](https://github.com/zake7749/Chatbot)
Mianbot 是採用樣板與檢索式模型搭建的聊天機器人，目前有兩種產生回覆的方式，專案仍在開發中:)
其一是以詞向量進行短語分類，針對分類的目標模組實現特徵抽取與記憶回覆功能，以進行多輪對話，匹配方式可參考Semantic Graph（目前仍在施工中 ΣΣΣ (」○ ω○ )／）。
其二除了天氣應答外，主要是以 PTT Gossiping 作為知識庫，透過文本相似度的比對取出與使用者輸入最相似的文章標題，再從推文集內挑選出最為可靠的回覆，程式內容及實驗過程請參見PTT-Chat_Generator。


[Conchylicultor/DeepQA](https://github.com/Conchylicultor/DeepQA) (Similar implementation to chatbots inikdom/neural-chatbot)
This work tries to reproduce the results of [A Neural Conversational Model](https://arxiv.org/abs/1506.05869) (aka the Google chatbot). It uses a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

[qhduan/ConversationalRobotDesign](https://github.com/qhduan/ConversationalRobotDesign)这个repo会记录我对 Conversational Robot 的理解、学习、研究、设计、实现的相关内容

[qhduan/Seq2Seq_Chatbot_QA](https://github.com/qhduan/Seq2Seq_Chatbot_QA)这个repo诞生比较早，那个时候tensorflow还没到1.0版本， 所以这个模型当时用的tf.contrib.seq2seq库，现在已经是tf.contrib.legacy_seq2seq了， 我想大家明白legacy的意思。这个repo的本身目的是学习与实现seq2seq的相关内容， 并不是一个完整的software，所以它除了学习和别人参考来说，就有各种各样的问题。

[qhduan/just_another_seq2seq](https://github.com/qhduan/just_another_seq2seq)
主要是从个人角度梳理了一下seq2seq的代码
加入了可选基本的CRF支持，loss和infer（还不确定对
加入了一些中文注释
相对于其他一些repo，bug可能会少一些
有些repo的实现在不同参数下会有问题：例如有些支持gru不支持lstm，有些不支持bidirectional，有些选择depth > 1的时候会有各种bug之类的，这些问题我都尽量修正了，虽然不保证实现肯定是对的
后续我可能会添加一些中文的例子，例如对联、古诗、闲聊、NER
根据本repo，我会整理一份seq2seq中间的各种trick和实现细节的坑
pretrained embedding support

[JayParks/tf-seq2seq](https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py)

[marsan-ma/tf_chatbot_seq2seq_antilm](https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm)

[Yoctol/seq2vec](https://github.com/Yoctol/seq2vec)
Turn sequence of words into a fix-length representation vector. This is a version to refactor all the seq2vec structures and use customed layers in yklz.

[bshao001/ChatLearner](https://github.com/bshao001/ChatLearner)A chatbot implemented in TensorFlow based on the new sequence to sequence (NMT) model, with certain rules seamlessly integrated.
