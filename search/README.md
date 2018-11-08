# Search Engine setup

## Galago installation

* Downlaod galago from link https://sourceforge.net/projects/lemur/files/lemur/galago-3.9/galago-3.9.tar.gz/download

* Extract galago using :
```
gzip -d galago-3.9.tar.gz 
tar -xzvf galago-3.9.tar.gz
```

## Preprocess data for search

Run script to convert reviews into document that the search engine can index

```
python process.py
```

## Index documents using galago command 

This will take a long time.

For now you can skip this step. An index for 5000 reviews is created as a sample and uploaded into github. So can check out search with GUI below

```
galago build --inputPath=trecText --indexPath=./index/ --fileType=trectext --nonStemmedPostings=true --stemmedPostings=true --stemmer+krovetz --corpus=true --tokenizer/fields+text
```

## Launch GUI search tool

```
galago search --port=8081 --index=./index/ --scorer=dirichlet --defaultTextPart=postings.krovetz
```

## Check out results in browser

http://127.0.1.1:8081

![Search demo](galago_demo.png "Yelp search demo")