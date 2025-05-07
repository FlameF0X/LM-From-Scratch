# Data Collection & Processing

In this section, we’ll walk through how to gather and process text data for building a language model from scratch. Data is the foundation upon which your model will learn the structure and nuances of language. Without high-quality, diverse data, even the best model architecture won’t perform well. Let's break down the key steps of data collection and preprocessing.

---

## 1. Data Collection

The first step is gathering a large and diverse dataset. A good corpus should represent the language (or domain) that the model will work with. Language models can benefit from large, varied datasets to capture diverse linguistic patterns.

### Sources of Data:

- **Web Crawls:** Websites like Common Crawl scrape billions of pages across the web and are often used as a base for pre-trained language models.
- **Books:** Datasets like Project Gutenberg or the BookCorpus dataset offer vast collections of text from various genres.
- **Wikipedia:** Provides a comprehensive dataset for general knowledge, often used in models like GPT and BERT.
- **News Articles & Blogs:** Domain-specific datasets like news corpora help models specialize in topics like current events, finance, etc.
- **Social Media & Forums:** Data from Reddit, Twitter, or Stack Exchange helps train models for more casual or technical language.
- **Academic Papers:** Datasets such as arXiv and PubMed are ideal for building models in specialized domains like science and medicine.

**Considerations:**
- **Data Size:** For pre-training, the more data, the better (aim for hundreds of gigabytes to terabytes of text).
- **Diversity:** A diverse dataset helps the model understand language in varied contexts (formal, informal, technical, etc.).
- **Language Balance:** Ensure that your corpus includes the target language in a well-balanced manner to avoid skewing the model towards one style or domain.

---

## 2. Data Cleaning & Normalization

Once you’ve gathered the data, it needs to be cleaned and preprocessed to make it suitable for training. Raw text data can be messy and inconsistent, and it’s important to standardize it.

### Key Tasks:

- **Removing Non-Text Content:** Strip away HTML tags, code, links, and other non-linguistic elements.
- **Unicode Normalization:** Normalize different character encodings (e.g., converting all text to UTF-8).
- **Text Normalization:** Convert text to lowercase (optional) and remove unnecessary punctuation or special characters.
- **Filtering Out Noisy Text:** Exclude pages with minimal content (e.g., boilerplate text, random symbols).
- **Language Detection:** Ensure that the text data is in the desired language(s), and exclude irrelevant languages.

### Example:
If you are using web-scraped data, removing HTML tags and JavaScript is essential to avoid training the model on non-relevant text, which could hurt performance.

---

## 3. Sentence Segmentation

Sentence segmentation is crucial for understanding the structure of the language. Proper segmentation helps the model learn the grammatical structure and flow of sentences.

- **Using a Sentence Tokenizer:** Tools like NLTK or SpaCy provide sentence tokenizers that break text into meaningful sentences.
- **Handling Abbreviations & Punctuation:** Special rules may be needed for handling abbreviations (e.g., "Dr." should not be split as a sentence-ending period).
  
### Example:
Text like:
```
"Hello! How are you? I hope you're doing well."
```
Should be split into:
```
["Hello!", "How are you?", "I hope you're doing well."]
```

---

## 4. Tokenization

Tokenization is the process of splitting text into smaller, more manageable units (tokens) that the model will understand. There are different strategies for tokenization, and the choice depends on the type of model you plan to train.

### Common Tokenization Methods:

- **Word Tokenization:** Splitting text into words (e.g., "I love NLP" → ["I", "love", "NLP"]).
- **Subword Tokenization:** This method splits words into smaller meaningful parts, especially useful for handling rare words (e.g., "unhappiness" → ["un", "happiness"]).
- **Character-Level Tokenization:** Each character is treated as a token (e.g., "hello" → ["h", "e", "l", "l", "o"]).

**Tokenization Algorithms:**
- **Byte Pair Encoding (BPE):** Splits words into subwords iteratively based on frequency.
- **WordPiece:** Similar to BPE, but typically used with masked language models like BERT.
- **Unigram Language Model:** Uses a probabilistic model to decide which subword units are best.

Choosing the appropriate tokenization strategy depends on your model type and the vocabulary size you're aiming for.

---

## 5. Chunking the Data into Training Samples

Once you have tokenized your text, the next step is to break it down into manageable chunks (training samples) that the model can process during training.

### Key Considerations:
- **Sequence Length:** Define the maximum sequence length (e.g., 512 tokens) for each sample. Longer sequences are typically split into multiple smaller sequences.
- **Overlapping Segments:** In some cases, using overlapping sequences (sliding window) helps the model learn more context.
- **Padding:** If sequences are shorter than the defined length, padding is added to make them consistent in size.

Example: If your corpus has the sentence:
```
"The cat sat on the mat."
```
And you’ve chosen a sequence length of 5, you might break it into:
```
["The", "cat", "sat", "on", "the"]
["cat", "sat", "on", "the", "mat"]
```

---

## 6. Data Augmentation (Optional)

Data augmentation can help improve the model's robustness by artificially increasing the diversity of the training data. Common techniques include:

- **Text Perturbation:** Minor changes in word order or synonym replacement.
- **Back-Translation:** Translating the text to another language and then back to the original.
- **Noise Injection:** Adding random noise (misspellings, typos) to simulate real-world variability.

**Note:** Data augmentation is not always necessary for all models but can be beneficial in low-resource scenarios.

---

## 7. Dataset Splitting

For model evaluation, you need to split your data into training, validation, and test sets. The general practice is:

- **Training Set (80-90%):** The bulk of the data is used for training the model.
- **Validation Set (5-10%):** Used during training to tune hyperparameters and monitor performance.
- **Test Set (5-10%):** Used after training to evaluate the final performance of the model.

---

## Conclusion

Data collection and processing are the foundational steps in building a language model from scratch. Proper preprocessing ensures that your model can learn meaningful patterns from the data, while effective tokenization and chunking improve training efficiency. By carefully collecting, cleaning, and segmenting your data, you set your model up for success during the pre-training phase.

---
### NEXT TOPIC >>> [Tokenization: Building the Vocabulary](tokenization.md)

