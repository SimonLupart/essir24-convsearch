#############################
# Usefull for methods 1 and 2
#############################
## TOPIOCQA conversation
wget -O raw_train.json https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json?download=1
wget -O raw_dev.json https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json?download=1

## splade index of the TOPIOCQA collection
mkdir topiocqa_index
cd topiocqa_index
wget -O index.tar.gz https://surfdrive.surf.nl/files/index.php/s/TV9RLEYQqXA2Z04/download
tar -xzvf index.tar.gz
rm index.tar.gz
cd ..

#############################
# Usefull for method 3
#############################
## TOPIOCQA collection
wget -O full_wiki_segments.tsv https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv?download=1

## run file from splade for mining negatives in the contrastive loss
wget -O run.json https://surfdrive.surf.nl/files/index.php/s/3YEhMBM7UhOVQrU/download
