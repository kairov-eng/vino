---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:2043
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) <!-- at revision 1427fd652930e4ba29e8149678df786c240d8825 -->
- **Maximum Sequence Length:** 160 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ['ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П', 'Мускат Отборный | Винодельня 78 | Белое | Мускат Янтарный | ВИНОДЕЛЬНЯ №78 | Мускат Отборный | ручной сбор'],
    ['ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П', 'Каберне Совиньон | Винодельня Бегильдеева | Красное | Каберне Совиньон | Nº5 | MOREWINE.RU | ВИНО КРАСНОЕ СУХОЕ | КАБЕРНЕ СОВИНЬОН | ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА'],
    ['ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П', 'Каберне Совиньон – Мерло | Винодельня Бегильдеева | Красное | Каберне Совиньон | ВИНО КРАСНОЕ СУХОЕ | КАБЕРНЕ СОВИНЬОН | МЕРЛО | ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА'],
    ['ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П', 'Красностоп Золотовский | Винодельня Молчанова | Красное | Красностоп Золотовский | Винодельня Молчановых | Красностоп | Золотовский | Вино сухое красное'],
    ['ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П', 'Мысхако игристое белое полусладкое | Мысхако | Белое | Совиньон Блан, Шардоне | 1869 | МЫСХАКО | ВИНОДЕЛЬНЯ | Игристое | ПОЛУСЛАДКОЕ БЕЛОЕ | ИГРИСТОЕ ВИНО'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'ПРОИЗВЕДЕНО В КРЫМУ | Крымский | ПОГРЕБОК | Мускат | Ркацители | БЕЛОЕ | ПОЛУСЛАДКОЕ | КРЫМСКИЙ ПОГРЕБОК | 20 | 14 | К | П',
    [
        'Мускат Отборный | Винодельня 78 | Белое | Мускат Янтарный | ВИНОДЕЛЬНЯ №78 | Мускат Отборный | ручной сбор',
        'Каберне Совиньон | Винодельня Бегильдеева | Красное | Каберне Совиньон | Nº5 | MOREWINE.RU | ВИНО КРАСНОЕ СУХОЕ | КАБЕРНЕ СОВИНЬОН | ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА',
        'Каберне Совиньон – Мерло | Винодельня Бегильдеева | Красное | Каберне Совиньон | ВИНО КРАСНОЕ СУХОЕ | КАБЕРНЕ СОВИНЬОН | МЕРЛО | ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА',
        'Красностоп Золотовский | Винодельня Молчанова | Красное | Красностоп Золотовский | Винодельня Молчановых | Красностоп | Золотовский | Вино сухое красное',
        'Мысхако игристое белое полусладкое | Мысхако | Белое | Совиньон Блан, Шардоне | 1869 | МЫСХАКО | ВИНОДЕЛЬНЯ | Игристое | ПОЛУСЛАДКОЕ БЕЛОЕ | ИГРИСТОЕ ВИНО',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 2,043 training samples
* Columns: <code>text1</code>, <code>text2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text1                                                                                          | text2                                                                                           | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                          | float                                                          |
  | details | <ul><li>min: 9 characters</li><li>mean: 78.58 characters</li><li>max: 312 characters</li></ul> | <ul><li>min: 38 characters</li><li>mean: 175.1 characters</li><li>max: 650 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.11</li><li>max: 1.0</li></ul> |
* Samples:
  | text1                                                                                                                                                   | text2                                                                                                                                                                                                                                                                     | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>TOP 100 WINES \| ВИНОДЕЛЬНЯ \| 2023 ГОДА \| FANAGORIA \| Cru Lermont \| RIESLING \| ВИНО БЕЛОЕ СУХОЕ 2024 \| Сенной \| ЗНМП \| РУЧНОЙ СБОР</code> | <code>Cru Lermont Рислинг \| Фанагория \| Белое \| Рислинг Рейнский \| FANAGORIA \| Cru Lermont \| RIESLING \| ВИНО БЕЛОЕ СУХОЕ \| Сенной \| ЗНМП</code>                                                                                                                  | <code>1.0</code> |
  | <code>TOP 100 WINES \| ВИНОДЕЛЬНЯ \| 2023 ГОДА \| FANAGORIA \| Cru Lermont \| RIESLING \| ВИНО БЕЛОЕ СУХОЕ 2024 \| Сенной \| ЗНМП \| РУЧНОЙ СБОР</code> | <code>Cru Lermont Pinot Noir \| Фанагория \| Красное \| Пино Нуар \| FANAGORIA \| Cru Lermont \| 2019 \| PINOT NOIR \| ВИНО КРАСНОЕ СУХОЕ \| РУЧНОЙ СБОР, ВЫДЕРЖАНО В БОЧКАХ \| Сенной \| С ЗАЩИЩЕННЫМ НАИМЕНОВАНИЕМ \| МЕСТА ПРОИСХОЖДЕНИЯ</code>                        | <code>0.0</code> |
  | <code>TOP 100 WINES \| ВИНОДЕЛЬНЯ \| 2023 ГОДА \| FANAGORIA \| Cru Lermont \| RIESLING \| ВИНО БЕЛОЕ СУХОЕ 2024 \| Сенной \| ЗНМП \| РУЧНОЙ СБОР</code> | <code>Cru Lermont Cabernet Sauvignon \| Фанагория \| Красное \| Каберне Совиньон \| FANAGORIA \| Cru Lermont \| 2020 \| CABERNET SAUVIGNON \| ВИНО КРАСНОЕ СУХОЕ \| РУЧНОЙ СБОР, ВЫДЕРЖАНО В БОЧКАХ \| Сенной \| С ЗАЩИЩЕННЫМ НАИМЕНОВАНИЕМ \| МЕСТА ПРОИСХОЖДЕНИЯ</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 1,021 evaluation samples
* Columns: <code>text1</code>, <code>text2</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text1                                                                                          | text2                                                                                            | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                         | string                                                                                           | float                                                          |
  | details | <ul><li>min: 4 characters</li><li>mean: 66.98 characters</li><li>max: 201 characters</li></ul> | <ul><li>min: 50 characters</li><li>mean: 161.85 characters</li><li>max: 867 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.05</li><li>max: 1.0</li></ul> |
* Samples:
  | text1                                                                                                                                              | text2                                                                                                                                                                                       | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>ПРОИЗВЕДЕНО В КРЫМУ \| Крымский \| ПОГРЕБОК \| Мускат \| Ркацители \| БЕЛОЕ \| ПОЛУСЛАДКОЕ \| КРЫМСКИЙ ПОГРЕБОК \| 20 \| 14 \| К \| П</code> | <code>Мускат Отборный \| Винодельня 78 \| Белое \| Мускат Янтарный \| ВИНОДЕЛЬНЯ №78 \| Мускат Отборный \| ручной сбор</code>                                                               | <code>1.0</code> |
  | <code>ПРОИЗВЕДЕНО В КРЫМУ \| Крымский \| ПОГРЕБОК \| Мускат \| Ркацители \| БЕЛОЕ \| ПОЛУСЛАДКОЕ \| КРЫМСКИЙ ПОГРЕБОК \| 20 \| 14 \| К \| П</code> | <code>Каберне Совиньон \| Винодельня Бегильдеева \| Красное \| Каберне Совиньон \| Nº5 \| MOREWINE.RU \| ВИНО КРАСНОЕ СУХОЕ \| КАБЕРНЕ СОВИНЬОН \| ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА</code> | <code>0.0</code> |
  | <code>ПРОИЗВЕДЕНО В КРЫМУ \| Крымский \| ПОГРЕБОК \| Мускат \| Ркацители \| БЕЛОЕ \| ПОЛУСЛАДКОЕ \| КРЫМСКИЙ ПОГРЕБОК \| 20 \| 14 \| К \| П</code> | <code>Каберне Совиньон – Мерло \| Винодельня Бегильдеева \| Красное \| Каберне Совиньон \| ВИНО КРАСНОЕ СУХОЕ \| КАБЕРНЕ СОВИНЬОН \| МЕРЛО \| ФЕРМЕРСКАЯ ВИНОДЕЛЬНЯ ВЕГИЛЬДЕЕВА</code>      | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `learning_rate`: 2e-05
- `weight_decay`: 0.01
- `warmup_ratio`: 0.1
- `use_cpu`: True
- `load_best_model_at_end`: True
- `dataloader_pin_memory`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3.0
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: True
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: False
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch   | Step    | Training Loss | Validation Loss |
|:-------:|:-------:|:-------------:|:---------------:|
| 0.3906  | 50      | 0.527         | -               |
| 0.7812  | 100     | 0.2636        | -               |
| **1.0** | **128** | **-**         | **0.1423**      |
| 1.1719  | 150     | 0.2027        | -               |
| 1.5625  | 200     | 0.2505        | -               |
| 1.9531  | 250     | 0.1877        | -               |
| 2.0     | 256     | -             | 0.2075          |
| 2.3438  | 300     | 0.1524        | -               |
| 2.7344  | 350     | 0.2331        | -               |
| 3.0     | 384     | -             | 0.2562          |

* The bold row denotes the saved checkpoint.

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 5.1.0
- Transformers: 4.56.2
- PyTorch: 2.14.0+cpu
- Accelerate: 1.10.1
- Datasets: 4.1.1
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->