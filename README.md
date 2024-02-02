# Multimodal Clinical Pseudo-notes for Emergency Department Prediction Tasks using Multiple Embedding Model for EHR (MEME)

![GitHub](https://img.shields.io/github/license/Simonlee711/MEME) [![arXiv](https://img.shields.io/badge/arXiv-2402.00160-brightgreen.svg)](https://arxiv.org/abs/2402.00160)

In this work, we introduce Multiple Embedding Model for EHR (MEME), an approach that views Electronic Health Records (EHR) as multimodal data. It uniquely represents tabular concepts like diagnoses and medications as structured natural language text using our "pseudo-notes" method. This approach allows us to effectively employ Large Language Models (LLMs) for individual EHR representation, proving beneficial in a variety of text-classification tasks. We demonstrate the effectiveness of MEME by applying it to diverse tasks within the Emergency Department across multiple hospital systems. Our findings show that MEME surpasses the performance of both single modality/embedding methods and traditional machine learning approaches, highlighting its effectiveness. Additionally, our tests on the model's generalizability reveal that training solely on the MIMIC-IV database does not guarantee effective application across different hospital institutions.



# Citing
```
@misc{lee2024multimodal,
      title={Multimodal Clinical Pseudo-notes for Emergency Department Prediction Tasks using Multiple Embedding Model for EHR (MEME)}, 
      author={Simon A. Lee and Sujay Jain and Alex Chen and Arabdha Biswas and Jennifer Fang and Akos Rudas and Jeffrey N. Chiang},
      year={2024},
      eprint={2402.00160},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
