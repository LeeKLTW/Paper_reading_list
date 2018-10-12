# Paper_reading_list
A list to record the paper I am reading.

## RNN

|Title|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|
Sutskever et al,Sequence to sequence learning with neural networks |2014| [pdf](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)||| [Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/Sutskever%20et%20al%2CSequence%20to%20sequence%20learning%20with%20neural%20networks.md)|

## Text(NLP,NLU)

### Chatbots
|Title|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|
| A Deep Reinforcement Learning Chatbot | _7 sep 2017_ | [arxiv](https://arxiv.org/pdf/1709.02349) |  | _None_ | |
| A Neural Conversational Model | _19 jun 2015_ | [arxiv](https://arxiv.org/pdf/1506.05869) | [inikdom/neural-chatbot](https://github.com/inikdom/neural-chatbot) | _None_ |[Read](https://github.com/LeeKLTW/Paper_reading_list/blob/master/A%20Neural%20Conversational%20Model.md)|

### Taskbots
|Title|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|
| Composite Task-Completion Dialogue Policy Learning via Hierarchical Deep Reinforcement Learning | _10 apr 2017_ | [arxiv](https://arxiv.org/pdf/1704.03084) | [MiuLab/TC-Bot](https://github.com/MiuLab/TC-Bot) | _None_ ||
| End-to-End Task-Completion Neural Dialogue Systems | _3 mar 2017_ | [arxiv](https://arxiv.org/pdf/1703.01008) | [MiuLab/TC-Bot](https://github.com/MiuLab/TC-Bot) | _None_ | |

### Question Answering
|Title|Date|Paper|Code|Labels|Status|
|---|---|---|---|---|---|
| IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models | _30 may 2017_ | [arxiv](https://arxiv.org/pdf/1705.10513) | [geek-ai/irgan](https://github.com/geek-ai/irgan) | ![state-of-the-art](https://img.shields.io/badge/label-State--of--the--art-red.svg) | |


# github
[zake7749/Chatbot](https://github.com/zake7749/Chatbot)
Mianbot 是採用樣板與檢索式模型搭建的聊天機器人，目前有兩種產生回覆的方式，專案仍在開發中:)
其一（左圖）是以詞向量進行短語分類，針對分類的目標模組實現特徵抽取與記憶回覆功能，以進行多輪對話，匹配方式可參考Semantic Graph（目前仍在施工中 ΣΣΣ (」○ ω○ )／）。
其二（右圖）除了天氣應答外，主要是以 PTT Gossiping 作為知識庫，透過文本相似度的比對取出與使用者輸入最相似的文章標題，再從推文集內挑選出最為可靠的回覆，程式內容及實驗過程請參見PTT-Chat_Generator。


[Conchylicultor/DeepQA](https://github.com/Conchylicultor/DeepQA) (Similar to chatbots)
This work tries to reproduce the results of [A Neural Conversational Model](https://arxiv.org/abs/1506.05869) (aka the Google chatbot). It uses a RNN (seq2seq model) for sentence predictions. It is done using python and TensorFlow.

