# GenTaxo
**GenTaxo** (pronounced “gen-tech-so”) is a generation-based approach for taxonomy completion. It enhances taxonomy completion by identifying the positions in existing taxonomies that need new concepts and generating the concept names.

This repo contain source code used for self-supervised taxonomy expansion method GenTaxo, published in KDD 2021. 

[Enhancing Taxonomy Completion with Concept Generation via Fusing Relational Representations](https://arxiv.org/pdf/2106.02974.pdf)

## Requirements and Installation
GenTaxo currently runs on Linux，Mac and Windows with following requirements
- PyTorch >= 1.2.0
- Python  >= 3.6.0
- DGL >= 0.4.0
  
A detailed dependencies list can be found in `requirements.txt` and can be installed by:

```
  pip install -r requirements.txt
```
## Quick Start
### Step 1.a: Organize your input taxonomy along with node features into the following 3 files

**1. <TAXONOMY_NAME>.terms**, each line represents one concept in the taxonomy, including its ID and surface name

```
taxon1_id \t taxon1_surface_name
taxon2_id \t taxon2_surface_name
taxon3_id \t taxon3_surface_name
...
```

**2. <TAXONOMY_NAME>.taxo**, each line represents one relation in the taxonomy, including the parent taxon ID and child taxon ID

```
parent_taxon1_id \t child_taxon1_id
parent_taxon2_id \t child_taxon2_id
parent_taxon3_id \t child_taxon3_id
...
```

**3. <TAXONOMY_NAME>.terms.<EMBED_SUFFIX>.embed**, the first line indicates the vocabulary size and embedding dimension, each of the following line represents one taxon with its pretrained embedding

```
<VOCAB_SIZE> <EMBED_DIM>
taxon1_id taxon1_embedding
taxon2_id taxon2_embedding
taxon3_id taxon3_embedding
...
```

The embedding file follows the gensim word2vec format.

### Step 1.b: Generate train/validation/test partition files

You can generate your desired train/validation/test parition files by creating another 3 separated files (named <TAXONOMY_NAME>.terms.train, <TAXONOMY_NAME>.terms.validation, as well as <TAXONOMY_NAME>.terms.test) and puting them in the same directory as the above three required files.

These three partition files are of the same format -- each line includes one taxon_id that appears in the above <TAXONOMY_NAME>.terms file.

### Step 2: Generate the binary dataset file

1. create a folder "./data/{DATASET_NAME}"
2. put the above three required files (as well as three optional partition files) in "./data/{DATASET_NAME}"
3. under this root directory, run

```
python generate_dataset_binary.py \
    --taxon_name <TAXONOMY_NAME> \
    --data_dir <DATASET_NAME> \
    --embed_suffix <EMBED_SUFFIX> \
    --existing_partition 1 \
    --partition_pattern internal \
```

### Running Taxonomy Completion Module

```
python ./Taxonomy_Completion_Module/train.py --config config_files/$DATASET/config.json
```


### Running Concept Name Generation Module

```
./Concept_Name_Generation/sh run.sh
```

### Infer 

```
python ./Taxonomy_Completion_Module/infer.py --resume <MODEL_CHECKPOINT.pth> --taxon <INPUT_TAXON_LIST.txt> --save <OUTPUT_RESULT.tsv> --device 0
```
  
## Reference
```
@article{zeng2021enhancing,
  title={Enhancing Taxonomy Completion with Concept Generation via Fusing Relational Representations},
  author={Zeng, Qingkai and Lin, Jinfeng and Yu, Wenhao and Cleland-Huang, Jane and Jiang, Meng},
  booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```

## License
GenTaxo use the [MIT License](LISCENSE). The license applies to the pre-trained models as well.


## Acknowledgements
The code is implemented based on [TMN](https://github.com/JieyuZ2/TMN) and [GraphWriter-DGL](https://github.com/QipengGuo/GraphWriter-DGL).

## Contact Us
Contact Qingkai Zeng (<qzeng@nd.edu>), if you have any questions.


