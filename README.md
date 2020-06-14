# Crosslingual Topic Modeling with WikiPDA

_Tiziano Piccardi, Robert West_

We present Wikipedia-based Polyglot Dirichlet Allocation (_WikiPDA_), a crosslingual topic model that learns to represent Wikipedia articles written in any language as distributions over a common set of language-independent topics. It leverages the fact that Wikipedia articles link to each other and are mapped to concepts in the Wikidata knowledge base, such that, when represented as bags of links, articles are inherently language-independent. WikiPDA works in two steps, by first densifying bags of links using matrix completion and then training a standard monolingual topic model. A human evaluation shows that WikiPDA produces more coherent topics than monolingual text-based LDA, thus offering crosslinguality at no cost. We demonstrate WikiPDA’s utility in two applications: a study of topical biases in 28 Wikipedia editions, and crosslingual supervised classification. Finally, we highlight WikiPDA’s capacity for zero-shot language transfer, where a model is reused for new languages without any fine-tuning.

<hr>

Content of this repository:

* [TopicsExtractionPipeline](TopicsExtractionPipeline): The code used to generate the dataset and train the models. Use the code in this folder to regenerate the dataset with updated Wikipedia versions or to add more languages in the traning set.
* [Analysis](Analysis): Complete analysis described in the paper and appendix.
