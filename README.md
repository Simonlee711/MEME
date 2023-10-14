# Multimodal Clinical pseudo-notes for EHR prediction

Electronic Health Records (EHR) are comprehensive databases containing multimodal information about a patient's health history. In recent years, inference models have been developed using either a singular EHR modality or concatenated inputs for various predictive downstream tasks. However, a significant limitation of current methods is their failure to harness the joint representation of different modalities, thereby missing out on the true potential of EHR data. To address this issue, we treat EHR as multimodal  by introducing an additional layer of self-attention to our concatenated embeddings, followed by a projection into a reduced space for inference. We achieve this by pretraining a RoBERTa model from scratch, utilizing various textual EHR modalities sourced from the MIMIC-IV database through Self-Supervised Learning with a Masked Language Modeling (MLM) objective. We then fine-tune our model with this newly added layer for predicting Emergency Department (ED) Disposition, as well as explore its potential for zero-shot learning from our pre-training efforts.

## TODO

- QLoRA (for optimization)
- PEFT (for optimization)
- Benchmark against many more tests
- Rewrite paper, caveats: Computational Power to process size (1, 3000+, 768)
- Pick a multiclass label problem

