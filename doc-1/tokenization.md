# Tokenization

Tokenization is the process of converting raw text into smaller, manageable units (tokens) that a language model can understand and process. These tokens are the basic building blocks of the model’s input, which is crucial for the model to learn meaningful patterns in language. This document dives into the theory and methods of tokenization, including how to choose the best strategy for your language model.

---

## 1. What is Tokenization?

At its core, tokenization is about breaking down text into units that represent the smallest meaningful elements. For example, in the sentence:

```
"The quick brown fox jumps over the lazy dog."
```

Tokenization might break it into:

- **Word Tokenization:** ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
- **Subword Tokenization:** ["The", "qu", "ick", "brown", "fox", "jum", "ps", "over", "the", "lazy", "dog"]
- **Character Tokenization:** ["T", "h", "e", "q", "u", "i", "c", "k", ...]

Each token can be thought of as a discrete unit that the model processes independently. The way text is split into tokens depends on the strategy used, and different strategies come with advantages and trade-offs.

---

## 2. Types of Tokenization

### 2.1 Word Tokenization

Word tokenization splits the text into individual words. This method is simple and works well for languages with clear word boundaries (like English), but it struggles with languages that don't have explicit word delimiters or with rare words.

Example:
- Sentence: "I love programming."
- Tokens: ["I", "love", "programming"]

#### Pros:
- Simple to implement
- Intuitive to humans

#### Cons:
- Vocabulary size grows rapidly
- Struggles with out-of-vocabulary (OOV) words

---

### 2.2 Subword Tokenization

Subword tokenization breaks words into smaller units, often based on character-level patterns or statistically frequent subword sequences. This method is especially useful for handling rare words or languages with complex morphology.

**Byte Pair Encoding (BPE)** and **WordPiece** are two common subword tokenization algorithms.

- **Byte Pair Encoding (BPE):** Starts with characters as the basic units, and then iteratively merges the most frequent pairs of characters into new subword units.
- **WordPiece:** Similar to BPE but uses a probabilistic approach to generate the most likely subword units.

Example:
- Sentence: "unhappiness"
- BPE Tokens: ["un", "happiness"]
- WordPiece Tokens: ["un", "happiness"]

#### Pros:
- Reduces vocabulary size
- Handles rare words and OOV words better
- More efficient than word-level tokenization

#### Cons:
- Can introduce unnecessary subword splitting in common words
- Requires more complex implementation

---

### 2.3 Character Tokenization

Character-level tokenization treats each character as a separate token. While this approach can capture fine-grained linguistic information, it results in much longer sequences and requires more training steps to learn meaningful patterns.

Example:
- Sentence: "hello"
- Tokens: ["h", "e", "l", "l", "o"]

#### Pros:
- Can handle any word, even unseen or made-up words
- Very flexible

#### Cons:
- Sequences become much longer (inefficient for long texts)
- Requires more data to learn meaningful patterns

---

## 3. Common Tokenization Methods

### 3.1 Byte Pair Encoding (BPE)

BPE is an unsupervised subword tokenization algorithm that splits words into subword units based on the frequency of consecutive character pairs in the text. The algorithm iteratively merges the most frequent pairs of characters into a single token.

Example:

```
Input: "low", "lowest", "newer", "wider"
Initial tokens: ["l", "o", "w", "l", "o", "w", "e", "s", "t", "n", "e", "w", "e", "r"]
After applying BPE: ["low", "est", "new", "er"]
```

#### Pros:
- Reduces vocabulary size
- Efficient for handling rare words
- Widely used in NLP models (e.g., GPT-2, BERT)

#### Cons:
- Sometimes generates tokens that are not linguistically meaningful
- May split frequently occurring words into suboptimal subwords

### 3.2 WordPiece

WordPiece is similar to BPE but uses a probabilistic approach to choose subword units based on maximum likelihood estimation (MLE). This method assigns a probability to each potential subword, allowing for a more refined choice of splits.

Example:
- Sentence: "unhappiness"
- WordPiece tokens: ["un", "##happiness"]

The "##" denotes that "happiness" is a continuation from the prefix "un".

#### Pros:
- Handles rare and OOV words effectively
- Better for multilingual models (e.g., BERT)

#### Cons:
- Needs a separate training phase to generate the vocabulary
- Can generate awkward token splits for common words

### 3.3 Unigram Language Model

The Unigram Language Model (used in models like SentencePiece) creates a vocabulary of subwords by modeling the likelihood of each token appearing in the dataset. It uses a probabilistic approach to select subword units based on the frequency of token occurrence.

Example:
- Sentence: "I love programming"
- Unigram tokens: ["I", "love", "pro", "gram", "ming"]

#### Pros:
- High-quality subword segmentation
- Good at handling multiple languages and domain-specific vocabularies

#### Cons:
- More complex to implement than BPE or WordPiece
- Can sometimes produce overly fine-grained splits

---

## 4. Vocabulary Size and Handling Out-of-Vocabulary (OOV) Tokens

One of the key challenges of tokenization is determining the vocabulary size, which directly impacts model performance. A large vocabulary allows the model to work with more specific tokens, while a small vocabulary forces the model to work with more general units, like subwords.

### Key Considerations:

- **Vocabulary Size:** Most models use a vocabulary between 30k and 50k tokens. The larger the vocabulary, the more specific tokens the model can handle, but it also leads to increased memory requirements.
- **OOV Tokens:** Tokens that are not in the vocabulary must be handled. This is where subword tokenization shines, as it can split OOV words into known subword units (e.g., "unhappiness" → ["un", "happiness"]).

---

## 5. Padding and Truncation

In many NLP tasks, sequences of text must be the same length. Padding ensures that all sequences in a batch have the same length, while truncation shortens sequences that exceed a set length.

### Key Concepts:
- **Padding:** Add special tokens (e.g., `[PAD]`) to sequences that are too short.
- **Truncation:** Cut sequences that are too long to fit within a model's input size limit (e.g., 512 tokens).

Example:
- Original sequence: ["I", "love", "programming"]
- Padded sequence (length = 5): ["I", "love", "programming", "[PAD]", "[PAD]"]

---

## 6. Choosing the Right Tokenizer

The choice of tokenizer depends on the language model’s design and the nature of the text you're working with. Here’s a quick guide to choosing the right method:

- **Use Word Tokenization** when the vocabulary is small, and the text is simple and consistent.
- **Use Subword Tokenization (BPE, WordPiece, or SentencePiece)** when handling rare words, large datasets, or multilingual corpora.
- **Use Character Tokenization** when you want to build a highly flexible model capable of handling any text (useful for tasks like text generation or OCR).

---

## Conclusion

Tokenization is a critical step in building an effective language model. The choice of tokenization method will significantly influence the model’s performance, computational efficiency, and ability to generalize to new text. By understanding the different methods and their trade-offs, you can make an informed decision about how best to preprocess your text data.

---
### NEXT TOPIC >>> [Model Architecture Design](model-architecture.md)
