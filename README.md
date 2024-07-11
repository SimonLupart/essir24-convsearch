# Neural Conversational Search
**Author:** Simon Lupart - IRLab@UvA

> This work was originally developed as one of the tutorials for the [ESSIR'24](https://2024.essir.eu/) (15th European Summer School on Information Retrieval) hackathon!

While traditional search tasks focus on finding relevant documents in response to a single query, conversational search (CS) extends this by incorporating dialogue between the user and the search system. CS thus include the resolution of ellipses, anaphora, and topic switches by leveraging entire conversational history. This repository explores three different methods to address this challenge:

1. **Adhoc Retrieval Model:** Utilizing a model trained on traditional IR tasks but applied in conversational contexts.
2. **Large Language Models (LLMs):** Employing Llama3 to rephrase and resolve ambiguities from the conversational context.
3. **Fine-tuned Retrieval Model:** Aligning a retrieval model specifically to the conversational dataset (TOPIOCQA).

All our experiments further rely on SPLADE models, which leverage learned sparse representations, and we explore the methods on one of the publicly available Conversational Search dataset, TOPIOCQA.

## Installation

### Conda Environment

```bash
conda env create -f environment.yml
conda activate cs
```

### Repository Organization

- **README.md**: Overview of the tutorial and intructions.
- **DATA/**: Dataset repository.
- **EXP/**: Exp output repository.
- **llm/**: Code to leverage LLMs (e.g. Llama3).
- **splade/**: Simplified version of the SPLADE model. More documentation can be found in `splade/README.md`.
- **SLURM/**: Scripts to run the different approaches.


## Data

This tutorial is based on the [TOPIOCQA](https://mcgill-nlp.github.io/topiocqa/) dataset. This dataset contains several thousand conversations with relevance from a 25M passages wikipedia collection. The particularity of TOPIOCQA is that authors included topics switches within the conversations to make it more realistic.

Download the dataset by navigating to the `DATA` directory and running `dl.sh`. Here's an example of the dataset:

```
Q1: when will the new dunkirk film be released on dvd
A1: 18 december 2017 [passage-id 14979903]

Q2: what is this film about
A2: Dunkirk evacuation of World War II [passage-id 14979882]

Q3: can you mention a few members of the cast
A3: Fionn Whitehead, Tom Glynn-Carney, Jack Lowden, Harry Styles [passage-id 14979882]
```

Each turn includes a question and an answer, with answers referencing relevant passages. The Conversational Search task aims to retrieve the appropriate passage-id based on previous turns, even when questions like "Q2: what is this film about" could be ambiguous without previous context.


## Methods

### First Method: Adhoc Retrieval


This method applies a state-of-the-art SPLADE model, trained on MSMARCO, to the conversational setting. Although MSMARCO is not designed for conversations, using this model with the entire conversation or just a subset provides baseline results. However, full conversation inputs can be too long, and using only the last utterance may lead to ambiguity.

- **Resources**: 
  - [SPLADE Model Information](https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/)

**TODO**:
- Parse the data using `DATA/parse_topiocqa.py`.
- Adjust `SLURM/run_splade_adhoc.sh` to use different inputs (last utterance or full conversation).
- Output will be displayed in `EXP/out_adhoc/`.
- Report and discuss the different results.

---

### Second Method: LLMs

To address the limitations of the first method, this approach uses LLMs to rewrite the last query into a self-contained form. LLMs like Llama3 can handle longer sequences (up to 4k tokens) compared to SPLADE's 256 tokens, thus ensuring the input length remains manageable.

For example, "Q2: what is this film about" can be rewritten as "what is the Dunkirk film about," which SPLADE can then process effectively.

**TODO 1**:
- Use the prompts in `llm/generate_llama3.py`.
- Rewrite each turn's last query into a non-ambiguous form.
- Save results in `llm/generated/*.tsv`.
- Run SPLADE retrieval with the rewriten queries with `SLURM/run_llama3.sh`.

Another option is for the LLM to generate an answer for the current question, e.g., for "Q2: what is this film about," the LLM might respond, "Dunkirk film is about the Second World War." This answer can then be used as input for the SPLADE model to retrieve relevant passages.

Note: In RAG (retrieval augmented generation), we first retrieve passages relevant to a question and then based on those passages
generate an answer. Here in this approach we do the opposite (generate and then retrieve). It has been shown that this kind of 
approach has good performance.

**TODO 2**:
- Reformulate the prompts in `llm/generate_llama3.py`.
- Answer the last user utterance with the LLM.
- Save results in `llm/generated/*.tsv`.
- Apply the SPLADE model on the LLM-generated answer, with the same `SLURM/run_llama3.sh` script.

While this approach is very effective, it still has some challenges, some of them are for example:
- Misalignment between the rewrite task and the retrieval task.
- Possible hallucinations in answer generation which lead to wrong retrieval.
- Additional latency from the rewriting task.

We thus propose the last method which finetune directly on the conversational dataset.

---

### Third Method: Fine-tuned CS

To fully align with the TOPIOCQA dataset, this method fine-tunes the SPLADE model on the dataset's training set using a contrastive loss (i.e. INFO-NCE [blog-post](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec)). For each conversation history, the positive passage is the gold-annotated document, and negatives are sampled from the top 1k passages from a previous run.

For this method we recreate a new conda env:
```bash
conda env create -f conda_splade_env.yml
conda activate splade
```

**TODO**:
- Training and Inference scripts: `SLURM/run_splade_conv.sh`.
- Code is included in the `splade/` directory. Further details are included in the `splade/README.md`.
- Output will be in `EXP/out_ft/`.

---

The END! If you have questions or find a bug, feel free to contact us (s.c.lupart@uva.nl).
