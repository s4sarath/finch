[Download Data](https://pan.baidu.com/s/1uXY2oY5s_EFSuQEznbvmBQ)

```
├── configs
│   └── rnn_config.py   
│
├── data               
│   └── dataloaders
│   	 └── dataloader_char_rnn.py
│   	 └── dataloader_word_fixed.py
│   	 └── dataloader_word_rnn.py
│   	 └── preprocess_char_rnn.py
│   	 └── preprocess_word_fixed.py
│   	 └── preprocess_word_rnn.py
│   └── files_original
│   	 └── char_embed.txt        # download and place here
│   	 └── question.csv          # download and place here
│   	 └── test.csv              # download and place here
│   	 └── train.csv             # download and place here
│   	 └── word_embed.txt        # download and place here
│   └── files_processed
│   └── notebooks
│   └── tfrecords
│
├── log             
│   └── example.py
│   
├── mains              
│   └── train_word_siamese_rnn.py  # run this
│  
└── model  
    └── siamese_rnn.py
```
