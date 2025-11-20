from transformers import WhisperForConditionalGeneration,Seq2SeqTrainingArguments,EarlyStoppingCallback,Seq2SeqTrainer,WhisperProcessor
import pickle
import humanize
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import json
import evaluate
from pathlib import Path
from torch.utils.data import DataLoader
import gc
import numpy as np
import random
import pandas as pd
import re
import jiwer
random.seed(42)

torch.cuda.set_device(0) 

def get_dataset_duration(data):
  """Calculate the duration of dataset in seconds"""
  dataset_duration = [example["input_length"] for example in data]
  return sum(dataset_duration)

def get_trainable_parameters(model):
  """
  count and print number of trainable paraameters in the model
  """
  trainable_params = 0
  for param in model.parameters():
    if param.requires_grad:
      trainable_params += param.numel()
  all_params = sum(p.numel() for p in model.parameters())
  print(
    f"Total Parameters: {humanize.intword(all_params)} | Trainable Parameters: {trainable_params/all_params *100:.2f}%")

  return  (trainable_params, all_params)


def load_normalizer(english_spelling_mapping_path=None):
  if not english_spelling_mapping_path:
    normalizer = EnglishTextNormalizer()
  else:
    with open(english_spelling_mapping_path, "r") as file:
      english_spelling_mapping = json.load(file)
    normalizer = EnglishTextNormalizer(english_spelling_mapping)
    return normalizer

def load_model(model_id,tie_word_embeddings=False):
  """Load vanilla whisper pre-trained mdoel"""
  model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda:0",
        tie_word_embeddings=tie_word_embeddings,
      )
  return model

def set_layer_trainability(model,target_modules):
  for name, param in model.named_parameters():
    if any(l in name for l in target_modules):
      param.requires_grad = True
    else:
      param.requires_grad = False

def load_pickle(path_to_pickle):
  with open(path_to_pickle,"rb") as file:
    loaded_pickle=pickle.load(file)
  return loaded_pickle

def remove_punctuation_except_contractions(text):
    # Regular expression pattern to match punctuation except apostrophes used in contractions
    pattern = r"(?!')\b[^\w\s']+|\s[^\w\s']+\b"
    # Replace matched punctuation with an empty string
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
  processor: Any

  def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    # split inputs and labels since they have to be of different lengths and need different padding methods
    # first treat the audio inputs by simply returning torch tensors
    input_features = [{"input_features": feature["input_features"]} for feature in features]
    batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

    # get the tokenized label sequences
    label_features = [{"input_ids": feature["labels"]} for feature in features]
    # pad the labels to max length
    labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    # if bos token is appended in previous tokenization step,
    # cut bos token here as it's append later anyways
    if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
      labels = labels[:, 1:]

    batch["labels"] = labels

    return batch

class NumpyEncoder(json.JSONEncoder):
  """serilize json for dump"""

  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return super(NumpyEncoder, self).default(obj)


def clear_cuda_memory():
  """Frees unused memory on the CUDA device and collects garbage."""
  torch.cuda.empty_cache()
  gc.collect()

# def select_percent_chunks_indexes(total_items, percent):
#   """Select indexes of chunks of the total items"""
#   chunck_size = math.ceil(total_items * (percent / 100.0))

#   index_chuncks = []

#   for i in range(0, total_items, chunck_size):
#     end_index = min(i + chunck_size, total_items)

#     index_chuncks.append(list(range(i, end_index)))
#   return index_chuncks

def create_random_sublists(original_list, num_sublists=10, sublist_frac_size=15):
    """Create random sublists from an original list. If sublist_frac_size it 
        ruturns one list including whole list. """

    sublists = [[] for _ in range(num_sublists)]

    selected_members = random.sample(original_list, num_sublists)

    for i, member in enumerate(selected_members):
        sublists[i].append(member)
    sublist_size = int(len(original_list) * (sublist_frac_size / 100))
    remaining_elements = [
        elem for elem in original_list if elem not in selected_members
    ]
    if sublist_frac_size == 100:
        for sublist in sublists:
            # sublists = [[i for i in range(len(original_list))]]
            sublist.extend([elem for elem in original_list if elem not in sublist])
    else:
        for sublist in sublists:
            sublist.extend(random.sample(remaining_elements, sublist_size - 1))

    return sublists

def calculate_epochs(dataset_length, fractions, total_epochs):
  """Calcuating necessary number of epochs for each fraction of data
      to have the same number of samples for training with each fraction of data"""
  epochs_dict = {}
  
  for fraction in fractions:
      subset_samples = dataset_length * fraction
      total_data_seen = dataset_length * total_epochs
      num_epochs_fraction = (total_data_seen / subset_samples)*100
      epochs_dict[fraction] = int(num_epochs_fraction)
  
  return epochs_dict

class ExperimentManager():

  def __init__(self,config):
    

    self.__model_id__ = config.get("model_id")
    self.__language__ = config.get("language","en")
    self.__task__ = config.get("task","transcribe")
    self.__tie_word_embeddings__ = config.get("tie_word_embeddings",False)
    self.__vectorized_data_path__ = config.get("vectorized_data_path")
    self.__EXP_ID__ = config.get("experiemnt_id")
    self.__target_modules__ = config.get("target_modules")
    self.__data_allocation_fraction__ = config.get("data_allocation_fraction",10)
    self.__validation_range__ = config.get("validation_range",100)
    # self.__step__ = config.get("step",5)
    self.__processor__ = WhisperProcessor.from_pretrained(self.__model_id__, language=self.__language__, task=self.__task__)
    self.__do_lower_case__ = config.get("do_lower_case",False)
    self.__do_normalize__ = config.get("do_normalize",True)
    self.__english_spelling_mapping_path__ = config.get("english_spelling_mapping_path",None)
    self.__normalizer__ = load_normalizer(self.__english_spelling_mapping_path__)

    self.__vectorized_data__ = load_pickle(self.__vectorized_data_path__)

    self.__metric__ = evaluate.load("wer")
    self.__base_model_wer__ = config.get("base_model_wer",31.90)
    self.__data_collector__ = DataCollatorSpeechSeq2SeqWithPadding(processor=self.__processor__)

    # training args
    self.__report_to__ = config.get("report_to",["tensorboard"])
    self.__per_device_train_batch_size__ = config.get("per_device_train_batch_size",16)
    self.__gradient_accumulation_steps__ = config.get("gradient_accumulation_steps",1)
    self.__learning_rate__ = config.get("learning_rate",1e-6)
    self.__weight_decay__ = config.get("weight_decay",0.0)
    self.__evaluation_strategy__ = config.get("evaluation_strategy","epoch")
    self.__save_strategy__ = config.get("save_strategy","epoch")
    self.__logging_strategy__ = config.get("logging_strategy","epoch")
    self.__total_num_train_epochs__ = config.get("total_num_train_epochs",10)
    self.__save_total_limit__ = config.get("save_total_limit",1)
    self.__fp16__ = config.get("fp16",True)
    self.__per_device_eval_batch_size__ = config.get("per_device_eval_batch_size",8)
    self.__generation_max_length__ = config.get("generation_max_length",225)
    self.__load_best_model_at_end__ = config.get("load_best_model_at_end",True)
    self.__metric_for_best_model__ = config.get("metric_for_best_model","wer")
    self.__greater_is_better__ = config.get("greater_is_better",False)
    self.__predict_with_generate__ = config.get("predict_with_generate",True)
    self.__optim__ = config.get("optim","adamw_torch")
    self.__disable_tqdm__ = config.get("disable_tqdm",True)
    # callback arguments
    self.__early_stopping_patience__ = config.get("early_stopping_patience",10)

    self.__wer_penalty__ = config.get("wer_penalty",0)
    self.__idx=0

  def compute_metrics(self,pred):
    """compute metric"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = self.__processor__.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = self.__processor__.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = self.__processor__.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * self.__metric__.compute(predictions=pred_str, references=label_str)
    # wil= 100 * jiwer.wil(reference = label_str, hypothesis = pred_str)
    df = pd.DataFrame()
    decoded_ref_all_flat = label_str
    decoded_pred_all_flat = pred_str

    #temp to write the output of evaluation
    df["ref"] = decoded_ref_all_flat
    df["pred"] = decoded_pred_all_flat
    
    df.to_csv(f"eval_eddac_wil/eval_{self.__idx}.csv")
    self.__idx +=1
    if self.__do_lower_case__:
      label_str = [label.lower() for label in label_str]
      pred_str = [pred.lower() for pred in pred_str]
    if self.__do_normalize__:
      label_str = [self.__normalizer__(label) for label in label_str]
      pred_str = [self.__normalizer__(pred) for pred in pred_str]
      # label_str = [remove_punctuation_except_contractions(label).strip() for label in label_str]
      # pred_str = [remove_punctuation_except_contractions(pred).strip() for pred in pred_str]

    wer_norm = 100 * self.__metric__.compute(predictions=pred_str, references=label_str)
    # wil_norm = 100 * jiwer.wil(reference = label_str, hypothesis = pred_str)

    return {
            "wer":wer,
            # "wil":wil,
            "wer_norm": wer_norm,
            # "wil_norm":wil_norm
            }

  def infrence_on_test_set(self):

    test_vectorized_data = self.__vectorized_data__["test"]
    dataloader = DataLoader(test_vectorized_data, batch_size=8, collate_fn=self.__data_collector__)
    forced_decoder_ids = self.__processor__.get_decoder_prompt_ids(language=self.__language__, task=self.__task__)

    self.model.eval()
    decoded_preds_all = []
    decoded_labels_all = []
    for step, batch in enumerate(dataloader):
      with torch.cuda.amp.autocast():
        with torch.no_grad():
          generated_tokens = (self.model.generate(
            input_features=batch["input_features"].to("cuda:0"),
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=self.__generation_max_length__,
          ).cpu().numpy())
          labels = batch["labels"].cpu().numpy()
          labels = np.where(labels != -100, labels, self.__processor__.tokenizer.pad_token_id)

          decoded_preds = self.__processor__.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
          decoded_preds_all.append(decoded_preds)

          decoded_labels = self.__processor__.tokenizer.batch_decode(labels, skip_special_tokens=True)
          decoded_labels_all.append(decoded_labels)
      del generated_tokens, labels, batch
      clear_cuda_memory()

      decoded_ref_all_flat = [label for item in decoded_labels_all for label in item]
      decoded_pred_all_flat = [pred for item in decoded_preds_all for pred in item]

      decoded_ref_all_flat = [self.__normalizer__(label) for label in decoded_ref_all_flat]
      decoded_pred_all_flat = [self.__normalizer__(pred).strip()
                              for pred in decoded_pred_all_flat]

      wer = 100 * self.__metric__.compute(predictions=decoded_pred_all_flat,
                                references=decoded_ref_all_flat)

      return wer


  def train(self):

    vectorized_data_train = self.__vectorized_data__["train"]
    vectorized_data_validation = self.__vectorized_data__["validation"]
    validation_indexes = int(self.__validation_range__ / 100 * len(vectorized_data_validation))
    validation_subset = vectorized_data_validation.select([i for i in range(validation_indexes)])
    validation_length = get_dataset_duration(validation_subset)

    epoch_for_fraction=calculate_epochs(len(vectorized_data_train),self.__data_allocation_fraction__,self.__total_num_train_epochs__)
    
    for data_frac in self.__data_allocation_fraction__:

      training_indexes_list = create_random_sublists([i for i in range(len(vectorized_data_train))],sublist_frac_size=data_frac)

      for idx,tr_idx in enumerate(training_indexes_list):
        training_subset = vectorized_data_train.select(tr_idx)
        train_length = get_dataset_duration(training_subset)
      
      
        self.model = load_model(self.__model_id__,self.__tie_word_embeddings__)
        if not self.__tie_word_embeddings__ :
          self.model.proj_out.weight.data.copy_(self.model.model.decoder.embed_tokens.weight.data)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.dropout = 0.2
        self.model.config.activation_dropout =0.2


        set_layer_trainability(self.model,self.__target_modules__)
        trainable_parameters, all_params = get_trainable_parameters(self.model)

        if self.__early_stopping_patience__:
          callbacks = [EarlyStoppingCallback(early_stopping_patience=self.__early_stopping_patience__)]
        else:
          callbacks = None

        output_dir = Path("./output_edacc_startify")
        output_dir.mkdir(exist_ok=True,parents=True)
        output_dir = f"{output_dir}/{self.__EXP_ID__}_{data_frac}_{idx+1:02d}"


        training_args = Seq2SeqTrainingArguments(
          output_dir=output_dir,
          per_device_train_batch_size=self.__per_device_train_batch_size__,
          gradient_accumulation_steps=self.__gradient_accumulation_steps__,  # increase by 2x for every 2x decrease in batch size
          learning_rate=self.__learning_rate__,
          weight_decay=self.__weight_decay__,
          evaluation_strategy="epoch",
          save_strategy="epoch",
          logging_strategy="epoch",
          warmup_steps=10,
          num_train_epochs=epoch_for_fraction[data_frac],
          gradient_checkpointing=False,
          ddp_find_unused_parameters = False,
          fp16=True,
          no_cuda=False,
          dataloader_pin_memory=True,
          per_device_eval_batch_size=self.__per_device_eval_batch_size__,
          predict_with_generate=self.__predict_with_generate__,
          generation_max_length=self.__generation_max_length__,

          report_to=["tensorboard"],
          load_best_model_at_end=self.__load_best_model_at_end__,
          metric_for_best_model=self.__metric_for_best_model__,
          greater_is_better=self.__greater_is_better__,
          push_to_hub=False,
          gradient_checkpointing_kwargs={"use_reentrant": False},
          save_total_limit=self.__save_total_limit__,
          optim="adamw_torch",
          disable_tqdm = self.__disable_tqdm__
      )

        trainer = Seq2SeqTrainer(
                    args=training_args,
                    model=self.model,
                    train_dataset=training_subset,
                    eval_dataset=validation_subset,
                    data_collator=self.__data_collector__,
                    compute_metrics=self.compute_metrics,
                    callbacks=callbacks,
                    tokenizer=self.__processor__,
                  )
                
        self.model.config.use_cache = False # silence the warnings. Please re-enable for inference!

        trainer.train()
        clear_cuda_memory()
        del self.model

        train_log = {
            "experiemnt_id":self.__EXP_ID__,
            "training_slice":idx+1,
            "train_length":train_length.item(),
            "validation_length":validation_length.item(),
            "data_allocation_fraction":data_frac,
            "all_params":all_params,
            "trainable_parameters":trainable_parameters,
          }

        with open(output_dir+"/log.json","w") as file:
          json.dump(train_log,file,cls=NumpyEncoder,indent=4)
      
      

      
    

    





  
