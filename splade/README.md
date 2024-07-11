# Tutorial: Neural Conversational Search - SPLADE

**Authors:** Simon Lupart - IRLab@UvA

---

This section of the code relies on the [SPLADE GitHub repository](https://github.com/naver/splade) from the original SPLADE authors, Formal et al. We only truncated all unnecessary functions in the context of this tutorial. This `README.md` file is specific to the SPLADE repository and outlines the organization of the various files contained within.

### Repository Overview

- **Training**:
  - The training of SPLADE models depends on HuggingFace code. It utilizes all methods and functions located in `/splade/hf_training/` and leverages the full hierarchy of HuggingFace libraries.

- **Indexing and Evaluation**:
  - Indexing and evaluation are performed using a Numba inverted index. All methods and functions for these tasks are defined in `/splade/evaluation/`.

**Note:** The model classes in `/splade/hf_training/` and `/splade/evaluation/` are different. This structure originates from the original code of the Naver repository. Therefore, if you modify the model architecture during training, ensure you also adjust the model architecture during inference to maintain consistency.

**Configuration**: The SPLADE repository utilizes Hydra for configuration management. You can find and modify the configuration files in `/splade/conf/`.

### Key Scripts

- **`hf_train.py`**: Script to train the model.
- **`retrieve.py`**: Script to perform retrieval.
- **`evaluate.py`**: Script to evaluate the model.


