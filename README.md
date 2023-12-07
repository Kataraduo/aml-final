# NexPost

## Datasets (from Kaggle)

- [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)
- [Instagram Posts](https://www.kaggle.com/datasets/thecoderenroute/instagram-posts-dataset/data)


## Repo Structure
 ```bash
nexpost/
│
├── data/             # Python scripts to process data
│   ├── flickr_dataset.py       #
│   ├── flickr8ks.py            #
│   ├── ig_posts.py             # convert raw ig_data to json
│   └── utils.py                #
│
├── datasets/         # Data only (raw + processed)
│   ├── flickr/
│   │   └── raw/
│   │       ├── Images/         # should contain all jpgs downloaded from Kaggle
│   │       └── captions.txt
│   ├── ig/
│   │   ├── raw/                # should contain all directories downloaded from Kaggle
│   │   └── ig_posts.json       # processed by ig_posts.py
│   └──
│
├── models/
│   ├──
│   └──
│
├── notebooks/
│   ├──
│   └──
│
├── outputs/
│   ├──
│   └──
│
├── ...
│
├── requirements.txt  # List of dependencies
├── settings.json
├── test.py
└── README.md
```
