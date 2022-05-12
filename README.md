# DL-Final-Project-2022
## Program Documentation

---

### Checkpoints
#### Dummy Dense
#### Testing Data
#### Testing Sparse

---
### Database
We filled our database directory by downloading a zip file already organized into gene/pathway/prostate
subdirectories and refactoring it into our local clone.
The link for the data is here (https://zenodo.org/record/5163213#.Ynb57hPMKlb)
---
### Data Pathways: reactome.py
P-NET is a feedforward neural network with constraints on nodes and edges. Each node encodes some biological entity (genes/pathways) and an edge represents a known relationship between the corresponding entities.

The constrains on the nodes allow a better understanding of the state of different biological components.
The constraints on the edges allow us to use a large number of nodes without increasing the number of edges, which leads to a smaller number of parameters compared to fully connected networks with the same number of nodes, and potentially fewer computations.

reactome.py processes the Reactome pathway datasets, found here: https://pubmed.ncbi.nlm.nih.gov/29145629/ (Reactome Pathway Knowledgebase)

The reactome dataset is downloaded and processed to form a layered network of five layers of pathways, one layer of genes, and one layer for features.
Spase model: 71,000 weights vs dense model: 270 million weights (first layer contains 94%). 
Hybrid model which contains a sparse layer followed by dense layers still contains >14 million weights.

`add_edges`: helper function to connect nodes

`complete_network`: consolidates subgraph, terminal nodes, edges and distances

`get_nodes_at_level`: get all nodes within distance around query node

`get_layers_from_net`: returns a list of layers with labeled nodes and successors

**Reactome class**: loads names, genes, and hierarchies

**ReactomeNetwork class**: gets terminals, roots, and Digraph hierarchy representation

---

### Data: gmt.reader.py

The P-NET model is not bound to a certain architecture, as model architecture is automatically built by reading model 
specifications provided by the user via a gene matrix transposed file format (.gmt file),
and custom pathways/gene sets/modules with custom hierarchies can be provided by the user,
making this a flexible tool.

**GMT class**: Loads architecture data from gmt, extracting a pathway column + gene column

---

### Layers: custom_layers.py

**Diagonal class**:
- Leverages keras to initialize/constrain bias/kernels/activation.
- Takes in a Layer to build it trainable weight variables

**SparseTF class**: 
- Same thing, but with a sparse architecture
- Additional option to build with random sparse constraints on the weights

---

### TODO: builder_utils.py
`get_map_from_layer`:

`get_layer_maps`:

`shuffle_genes_map`:

---

### config_path.py
Joins/organizes data/base paths

---

### TODO: data_reader.py
Introduction

Assembles patient profiles (molecular readouts) and  responses (labeled yes/no for cancer)

`load_data`: loads patient molecular profile data into pandas dataframes
- Option to take input that selects only certain genes from patient data
- Returns
  - patient profiles with selected genes (all if unspecified) and no labels with shape [num_patients, num_genes]
  - patient classification label (primary/metastatic cancer) - _UNCLEAR [num_patients, 1] (PANDAS?)_
  - patient barcode IDs [num_patients]
  - set of genes present in both selected genes and the dataset

`load_TMB`: finds the TMB in each patient sample, recorded in logs

Tumor Mutational Burden (TMB) is a relevant biomarker, 
as tumors with a high number of mutations are sometimes more responsive to immunotherapy,
as their cells are more likely to be recognized as abnormal and attacked by the immune system.
Consequently, understanding TMB may yield significant clinical insights.

`load_CNV`: finds the CNV burden in each patient sample, recorded in logs

CNV (Copy Number Variants) describe stretches of the genome where large sequences
have been abnormally duplicated or deleted.
Often, especially in genic regions, these CNVs may be implicated in cancer pathogenesis.
Therefore, CNVs are an important biomarker and worth recording.

`load_data_type`, `combine`, `split_cnv`: Preparation utility helper functions

**ProstateDataPaper class**: Master wrapper/loader for patient data
- loads patient information by data type: examples include TMB, various CNV types, various mutation/mutation hotspot types
- input option for only running selected genes/samples
- input option for shuffling loaded data
- `get_train_validate_test`: uses set intersection to return desired training/validation/testing sample splits

---

### dense.py

Dense model using keras.
- Adam optimizer
- Learning rate 0.01 
- Dense layer activation structure (16 tanh) ==> (4 tanh) ==> (2 softmax)
- Loss: BinaryCrossentropy

---

### TODO: run.py

`train_model`: 
