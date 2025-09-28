import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments
import os
import csv
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset, load_dataset, Audio

MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

def generate_dataset_from_csv(label_csv: str, data_folder: str) -> Dataset:
    dataset = load_dataset('csv', data_files=label_csv)

    dataset = dataset.map(lambda x: {"audio": os.path.join(data_folder, x["filename"])})

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

def generate_csv(data_folder: str, output_csv: str, overwrite: bool = False):
    if os.path.exists(output_csv) and not overwrite:
        print(f"{output_csv} already exists. Use overwrite=True to regenerate.")
        return
    
    writer = csv.writer(open(output_csv, 'w', newline=''))
    writer.writerow(['filename', 'label'])  # header
    files_processed = 0
    for fname in os.listdir(data_folder):
        if fname.endswith('.mp3'):
            # Extract phoneme+tone from filename
            base = os.path.splitext(fname)[0]
            phoneme_tone = base.split('_')[0]  # Adjust to your naming scheme

            writer.writerow([fname, phoneme_tone])
            files_processed += 1
    print(f"Generated {output_csv} with {files_processed} entries.")

def generate_vocab(label_csv: str, output_vocab: str = "datasets/tonevocab.json", overwrite: bool = False):
    if os.path.exists(output_vocab) and not overwrite:
        print(f"{output_vocab} already exists. Use overwrite=True to regenerate.")
        return

    df = pd.read_csv(label_csv)
    # Unique labels, sorted for consistency
    unique_labels = sorted(df["label"].unique())

    # Map each label to an ID
    vocab_dict = {label: idx for idx, label in enumerate(unique_labels)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # Save to JSON
    with open(output_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

# wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = 0 # wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

if __name__ == "__main__":
    # Example usage
    data_folder = "datasets/toneperfect"
    output_csv = "datasets/tonelabels.csv"
    vocab_json = "datasets/tonevocab.json"
    generate_csv(data_folder, output_csv)
    generate_vocab(output_csv, vocab_json)
    dataset = generate_dataset_from_csv(output_csv, data_folder)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    print(dataset["train"][0]["audio"])

    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    tokenizer = Wav2Vec2CTCTokenizer(vocab_json, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token=" ")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        with processor.as_target_processor():
            batch["labels"] = processor(batch["label"]).input_ids
        return batch
    
    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
    print(dataset)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    print(f"Original model vocab size: {model.config.vocab_size}")
    print(f"Custom tokenizer vocab size: {len(processor.tokenizer)}")

    # Replace the classification head to match your custom vocabulary
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, len(processor.tokenizer))

    # Update the model config to reflect the new vocab size
    model.config.vocab_size = len(processor.tokenizer)

    print(f"Updated model vocab size: {model.config.vocab_size}")

    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="finetuned-wav2vec2-tone",
        group_by_length=True,
        per_device_train_batch_size=32,
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    """
    dataset = TonePhonemeDataset(output_csv, data_folder, processor=processor)

    # Split into train/eval (e.g. 80/20)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_eval = n_total - n_train
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [n_train, n_eval])

    # Number of labels = number of phoneme+tone combos
    num_labels = len(dataset.label2id)

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        gradient_checkpointing=True, 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        #vocab_size=len(processor.tokenizer)
    )

    # Replace classification head to match vocab
    model.lm_head = torch.nn.Linear(model.lm_head.in_features, len(processor.tokenizer))

    model.freeze_feature_extractor()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        # output_dir=model_dir,
        output_dir="./wav2vec2-base-timit-demo",
        # output_dir="./wav2vec2-large-xlsr-turkish-demo",
        group_by_length=True,
        per_device_train_batch_size=16,
        #   per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        # evaluation_strategy="steps",
        num_train_epochs=50,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=3e-4,
        warmup_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,  # processor has both tokenizer & feature_extractor
    )

    trainer.train()
    """
