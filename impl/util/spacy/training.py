import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


def train_ner_model(training_data_func, output_dir: str, model='en', n_iter=50):
    # load or initialize model
    nlp = spacy.load(model)
    if 'ner' not in nlp.pipe_names:
        nlp.add_pipe('ner', last=True)

    # retrieve training data
    training_data = training_data_func(nlp)

    # train new model
    with nlp.select_pipes(enable='ner'):  # only train NER
        if model is None:
            nlp.initialize(lambda: training_data)  # initialize training if necessary (+ provide NE labels via examples)
        for itn in range(n_iter):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.5, losses=losses)

    # save model to output directory
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
