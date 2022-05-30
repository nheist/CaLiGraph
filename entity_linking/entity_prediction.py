from typing import Dict, List, Tuple, Set, Optional
import random
import utils
from impl.util.rdf import EntityIndex
from impl import subject_entity
from impl.subject_entity import combine
from impl.subject_entity.preprocess.word_tokenize import WordTokenizer, WordTokenizerSpecialToken
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModel
from impl.dbpedia.resource import DbpResource, DbpResourceStore
from entity_linking.preprocessing.embeddings import EntityIndexToEmbeddingMapper
from entity_linking.model.entity_prediction import TransformerForEntityPrediction
from entity_linking.data.entity_prediction import prepare_dataset
from entity_linking.evaluation.entity_prediction import EntityPredictionEvaluator


APPROACH = 'EPv2'


def run_prediction(model_name: str, sample: int, epochs: int, batch_size: int, loss: str, learning_rate: float, warmup_steps: int, weight_decay: float, num_ents: int, ent_dim: int, items_per_chunk: int, cls_predictor: bool, include_source_page: bool):
    run_id = f'{APPROACH}_{model_name}_s-{sample}_ipc-{items_per_chunk}_ne-{num_ents}_isp-{include_source_page}_cp-{cls_predictor}_ed-{ent_dim}_e-{epochs}_bs-{batch_size}_loss-{loss}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}'
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(WordTokenizerSpecialToken.all_tokens()))
    encoder = AutoModel.from_pretrained(model_name)
    encoder.resize_token_embeddings(len(tokenizer))
    ent_idx2emb = EntityIndexToEmbeddingMapper(ent_dim)
    model = TransformerForEntityPrediction(encoder, include_source_page, cls_predictor, ent_idx2emb, ent_dim, num_ents, loss)
    # load data
    dataset_version = f'{APPROACH}-{model_name}-s{sample}-ipc{items_per_chunk}-ne{num_ents}'
    train_data, val_data = utils.load_or_create_cache('vector_prediction_training_data', lambda: _load_train_and_val_datasets(tokenizer, sample, items_per_chunk, num_ents), version=dataset_version)
    # run evaluation
    training_args = TrainingArguments(
        seed=42,
        save_strategy=IntervalStrategy.NO,
        output_dir=f'./ep_eval/output/{run_id}',
        logging_strategy=IntervalStrategy.STEPS,
        logging_dir=f'./ep_eval/logs/{run_id}',
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
        compute_metrics=EntityPredictionEvaluator(ent_idx2emb, batch_size).evaluate
    )
    trainer.train()


def _load_train_and_val_datasets(tokenizer, sample: int, items_per_chunk: int, num_ents: int) -> Tuple[Dict[DbpResource, tuple], Dict[DbpResource, tuple]]:
    valid_res_indices = set(DbpResourceStore.instance().get_embedding_vectors())
    subject_entity_pages = combine.get_subject_entity_page_content(subject_entity._get_subject_entity_predictions())
    # filter out pages whose main entity has no embedding
    subject_entity_pages = {res: content for res, content in subject_entity_pages.items() if res.idx in valid_res_indices}
    # filter out listings of pages that have no labeled entities at all (because they violate the distant supervision assumption)
    subject_entity_pages = {res: _filter_listings_without_known_ents(content) for res, content in subject_entity_pages.items()}
    subject_entity_pages = {res: content for res, content in subject_entity_pages.items() if content is not None}  # ignore pages that have no more entities
    # split into train and validation
    sample_fraction = sample / 100  # sample is given as a percentage
    all_pages = random.sample(list(subject_entity_pages), int(len(subject_entity_pages) * sample_fraction))  # only use `sample` % of overall data

    train_pages = set(random.sample(all_pages, int(len(all_pages) * .9)))  # 90% of pages for train, 10% for val
    train_data = _create_vector_prediction_data({res: subject_entity_pages[res] for res in train_pages}, valid_res_indices, items_per_chunk, False)
    train_data = prepare_dataset(train_data, tokenizer, num_ents)

    val_pages = set(all_pages).difference(train_pages)
    val_data = _create_vector_prediction_data({res: subject_entity_pages[res] for res in val_pages}, valid_res_indices, items_per_chunk, True)
    val_data = prepare_dataset(val_data, tokenizer, num_ents)
    return train_data, val_data


def _filter_listings_without_known_ents(page_content: dict) -> Optional[dict]:
    has_ents = False
    for s in page_content['sections']:
        valid_enums = [enum for enum in s['enums'] if any('subject_entity' in entry and entry['subject_entity'] != EntityIndex.NEW_ENTITY for entry in enum)]
        s['enums'] = valid_enums
        valid_tables = [t for t in s['tables'] if any('subject_entity' in cell and cell['subject_entity'] != EntityIndex.NEW_ENTITY for row in t['data'] for cell in row)]
        s['tables'] = valid_tables
        if valid_enums or valid_tables:
            has_ents = True
    return page_content if has_ents else None


def _create_vector_prediction_data(subject_entity_pages: Dict[DbpResource, dict], valid_res_indices: Set[int], items_per_chunk: int, include_new_entities: bool) -> Dict[DbpResource, Tuple[List[List[str]], List[List[str]], List[List[int]]]]:
    entity_labels = _get_subject_entity_labels(subject_entity_pages, valid_res_indices, include_new_entities)
    return WordTokenizer(max_items_per_chunk=items_per_chunk, max_ents_per_item=1)(subject_entity_pages, entity_labels=entity_labels)


def _get_subject_entity_labels(subject_entity_pages: Dict[DbpResource, dict], valid_res_indices: Set[int], include_new_entities: bool) -> Dict[DbpResource, Tuple[Set[int], Set[int]]]:
    entity_labels = {}
    for res, page_content in subject_entity_pages.items():
        # collect all subject entity labels
        subject_entity_indices = set()
        subject_entity_indices.update({entry['subject_entity']['idx'] for s in page_content['sections'] for enum in s['enums'] for entry in enum if 'subject_entity' in entry})
        subject_entity_indices.update({cell['subject_entity']['idx'] for s in page_content['sections'] for table in s['tables'] for row in table['data'] for cell in row if 'subject_entity' in cell})
        # get rid of non-entities and entities without RDF2vec embeddings (as we can't use them for training/eval)
        subject_entity_indices = subject_entity_indices.intersection(valid_res_indices)
        if include_new_entities:
            subject_entity_indices.add(EntityIndex.NEW_ENTITY.value)
        entity_labels[res] = (subject_entity_indices, set())
    return entity_labels
