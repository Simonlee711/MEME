from transformers import TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenize
model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class Tokenizer:
    def tokenize(self, examples):
        """Mapping function to tokenize the sentences passed with truncation"""
        return tokenizer(
            examples["headline"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True,
        )

    def convert(self, l):
        """
        Run this method
        """
        arrival_hf = Dataset.from_pandas(l[0])
        triage_hf = Dataset.from_pandas(l[1])
        medrecon_hf = Dataset.from_pandas(l[2])
        vitals_hf = Dataset.from_pandas(l[3])
        codes_hf = Dataset.from_pandas(l[4])
        pyxis_hf = Dataset.from_pandas(l[5])

        arrival = arrival_hf.map(self.tokenize, batched=True)
        triage = triage_hf.map(self.tokenize, batched=True)
        medrecon = medrecon_hf.map(self.tokenize, batched=True)
        vitals = vitals_hf.map(self.tokenize, batched=True)
        codes = codes_hf.map(self.tokenize, batched=True)
        pyxis = pyxis_hf.map(self.tokenize, batched=True)

        print(arrival)

        arrival.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        triage.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        medrecon.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        vitals.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        codes.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        pyxis.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        return arrival, triage, medrecon, vitals, codes, pyxis
