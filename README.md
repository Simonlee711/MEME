# GeneratEHR: Multimodal Clinical pseudo-notes for EHR prediction

Electronic Health Records (EHR) are comprehensive databases containing multimodal information about a patient's health history. In recent years, Transformer-based models have shown promise in various downstream tasks, including mortality prediction and diagnosis. However, these approaches have either considered one component of EHR, or considered its multiple components as a single data modality. In this work, we treat EHR as multimodal, separately representing concepts like diagnoses, medications, procedures, and lab values. Our novel "pseudo-notes" method transforms these modalities into structured language texts, allowing us to leverage general Large Language Models (LLMs) for individual EHR representation from the MIMIC-IV database. Additionally, we introduce an additional self-attention layer for late fusion of these embeddings to gain a joint representation of a patient, followed by a projection into a reduced space for inference. We fine-tune our model with this newly added layer for predicting Emergency Department (ED) Disposition and find our multimodal model outperforms against a single modality method, and other machine learning methods, demonstrating its effectiveness.

## Authors
- **Simon Lee** (simonlee711@g.ucla.edu)
- Sujay Jain
- Alex Chen
- Akos Rudas (akosrudas@g.ucla.edu)
- Jeffrey Chiang (njchiang@g.ucla.edu)

