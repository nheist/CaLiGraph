from typing import Tuple
import utils
from impl.subject_entity.preprocess.word_tokenize import WordTokenizedSpecialToken
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModel
from entity_linking.data import prepare
from entity_linking.data.mention_entity_matching import prepare_dataset, MentionEntityMatchingDataset
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper
from entity_linking.model.mention_entity_crossencoder import MentionEntityCrossEncoder
from entity_linking.evaluation.mention_entity_matching import MentionEntityMatchingEvaluator


def run_prediction(version: str, model_name: str, epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float, num_ents: int, ent_dim: int, items_per_chunk: int, cls_predictor: bool, include_source_page: bool):
    run_id = f'MEMv{version}_{model_name}_ipc-{items_per_chunk}_ne-{num_ents}_isp-{include_source_page}_cp-{cls_predictor}_ed-{ent_dim}_e-{epochs}_bs-{batch_size}-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(WordTokenizedSpecialToken.all_tokens()))
    encoder = AutoModel.from_pretrained(model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    ent_idx2emb = EntityIndexToEmbeddingMapper(ent_dim)
    model = MentionEntityCrossEncoder(encoder, include_source_page, cls_predictor, ent_idx2emb, ent_dim, num_ents)
    # load data
    dataset_version = f'MEMv{version}-{model_name}-ipc{items_per_chunk}-ne{num_ents}'
    train_data, val_data = utils.load_or_create_cache('entity_linking_training_data', lambda: _load_train_and_val_datasets(tokenizer, items_per_chunk, num_ents), version=dataset_version)
    # run evaluation
    training_args = TrainingArguments(
        seed=42,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./entity_linking/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./entity_linking/logs/{run_id}',
        logging_steps=1000,
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=5000,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=MentionEntityMatchingEvaluator(batch_size, num_ents, items_per_chunk).evaluate
    )
    trainer.train()


def _load_train_and_val_datasets(tokenizer, items_per_chunk: int, num_ents: int) -> Tuple[MentionEntityMatchingDataset, MentionEntityMatchingDataset]:
    train_data, val_data = prepare.get_mem_train_and_val_data(items_per_chunk)
    train_dataset = prepare_dataset(train_data, tokenizer, num_ents, items_per_chunk)
    val_dataset = prepare_dataset(val_data, tokenizer, num_ents, items_per_chunk)
    return train_dataset, val_dataset
