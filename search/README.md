# Search Engine setup

## Galago installation

* Downlaod galago from link https://sourceforge.net/projects/lemur/files/lemur/galago-3.9/galago-3.9.tar.gz/download

* Extract galago using :
** gzip -d galago-3.9.tar.gz 
** tar -xzvf galago-3.9.tar.gz

## Run script to convert reviews into document that the search engine can index

python process.py

## Index the documents using galago command (This will take a long time)

galago build --inputPath=trecText --indexPath=./index/ --fileType=trectext --nonStemmedPostings=true --stemmedPostings=true --stemmer+krovetz --corpus=true --tokenizer/fields+text

## Launch GUI search tool

galago search --port=8081 --index=./index/ --scorer=dirichlet --defaultTextPart=postings.krovetz