from torch.utils.tensorboard import SummaryWriter
import os
import utils
from transformers import Trainer, IntervalStrategy, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification
from impl.util.transformer import SpecialToken
from impl.subject_entity.preprocess.pos_label import POSLabel
from entity_linking.data import prepare
from entity_linking.data.mention_detection import prepare_dataset
from entity_linking.model.mention_detection import TransformerForMentionDetectionAndTypePrediction
from entity_linking.evaluation.mention_detection import SETagsEvaluator


def run_evaluation(model_name: str, epochs: int, batch_size: int, learning_rate: float, warmup_steps: int, weight_decay: float, ignore_tags: bool, predict_single_tag: bool, negative_sample_size: float, single_item_chunks: bool, train_on_listpages: bool, train_on_pages: bool, save_as: str):
    run_id = f'v4_{model_name}_tlp-{train_on_pages}_tp-{train_on_pages}_it-{ignore_tags}_st-{predict_single_tag}_nss-{negative_sample_size}_sic-{single_item_chunks}_e-{epochs}_lr-{learning_rate}_ws-{warmup_steps}_wd-{weight_decay}'
    # check whether to use a local model or a predefined one from huggingface
    local_path_to_model = f'./entity_linking/MD/models/{model_name}'
    model_name = local_path_to_model if os.path.isdir(local_path_to_model) else model_name
    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, additional_special_tokens=list(SpecialToken.all_tokens()))
    number_of_labels = 2 if ignore_tags else len(POSLabel)
    if predict_single_tag:
        model = TransformerForMentionDetectionAndTypePrediction(model_name, len(tokenizer), number_of_labels)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=number_of_labels)
        model.resize_token_embeddings(len(tokenizer))

    if train_on_listpages:
        # load data
        train_data, val_data = utils.load_or_create_cache('MD_listpage_data', prepare.get_md_listpage_data)
        lp_train_dataset = prepare_dataset(train_data, tokenizer, ignore_tags, predict_single_tag, single_item_chunks, negative_sample_size)
        lp_val_dataset = prepare_dataset(val_data, tokenizer, ignore_tags, predict_single_tag, single_item_chunks)

        # run evaluation
        training_args = TrainingArguments(
            seed=42,
            save_strategy=IntervalStrategy.NO,
            output_dir=f'./entity_linking/MD/output/LP_{run_id}',
            logging_strategy=IntervalStrategy.STEPS,
            logging_dir=f'./entity_linking/MD/logs/LP_{run_id}',
            logging_steps=500,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=3000,
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
            train_dataset=lp_train_dataset,
            eval_dataset=lp_val_dataset,
            compute_metrics=lambda eval_prediction: SETagsEvaluator(eval_prediction, lp_val_dataset.listing_types, predict_single_tag).evaluate()
        )
        trainer.train()

    if train_on_pages:
        # load data
        train_data = utils.load_or_create_cache('MD_page_train_data', prepare.get_md_page_train_data)
        p_train_dataset = prepare_dataset(train_data, tokenizer, ignore_tags, predict_single_tag, single_item_chunks, negative_sample_size)

        # run evaluation
        training_args = TrainingArguments(
            seed=42,
            save_strategy=IntervalStrategy.NO,
            output_dir=f'./entity_linking/MD/output/P_{run_id}',
            logging_strategy=IntervalStrategy.STEPS,
            logging_dir=f'./entity_linking/MD/logs/P_{run_id}',
            logging_steps=500,
            evaluation_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=p_train_dataset)
        trainer.train()

    # make test predictions
    test_data = utils.load_or_create_cache('MD_page_test_data', prepare.get_md_page_test_data)
    test_dataset = prepare_dataset(test_data, tokenizer, ignore_tags, predict_single_tag, single_item_chunks)
    trainer = Trainer(model=model, compute_metrics=lambda eval_prediction: SETagsEvaluator(eval_prediction, test_dataset.listing_types, predict_single_tag).evaluate())
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')
    tb = SummaryWriter(log_dir=f'./entity_linking/MD/logs/{run_id}')
    for key, val in test_metrics.items():
        tb.add_scalar(f'test/{key[5:]}', val)

    if save_as is not None:
        path_to_model = f'./entity_linking/MD/models/{save_as}'
        model.save_pretrained(path_to_model)
        tokenizer.save_pretrained(path_to_model)
