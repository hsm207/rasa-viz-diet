import pathlib

import altair as alt
import numpy as np
import pandas as pd
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.nlu.model import Interpreter
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from sklearn.decomposition import PCA
from rasa.shared.core.domain import Domain


def _load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return Interpreter.load(nlu_model)


def create_interpreter(model_dir: str, model_name: str) -> Interpreter:
    return _load_interpreter(model_dir, f"{model_name}.tar.gz")


def _create_message(text, interpreter):
    data = interpreter.default_output_attributes()
    data[TEXT] = text
    return Message(data=data)


def _process_message(message, interpreter):
    for e in interpreter.pipeline:
        e.process(message)


def _extract_diet_intent_features(message):
    m = message.as_dict()[DIAGNOSTIC_DATA]

    # assume only 1 component with diagnostics data in the pipeline
    assert len(m) == 1

    for _, diagnostic_data in m.items():
        text_transformed = diagnostic_data["text_transformed"]

    return text_transformed[0]


def extract_cls_features(utterance: str, interpreter: Interpreter) -> np.ndarray:
    message = _create_message(utterance, interpreter)
    _process_message(message, interpreter)

    # The value of the __CLS__ token can be retrieved by taking the last token’s vector from text_transformed
    return _extract_diet_intent_features(message)[-1]


async def get_nlu_data(data_dir: str, domain_path: str, interpreter: Interpreter):
    training_data = TrainingDataImporter.load_from_dict(
        training_data_paths=[data_dir],
        domain_path=domain_path,
    )
    nlu_data = (await training_data.get_nlu_data()).intent_examples
    nlu_data = [
        (
            m.data["text"],
            m.data["intent"],
            extract_cls_features(m.data["text"], interpreter),
        )
        for m in nlu_data
    ]
    return pd.DataFrame(nlu_data, columns=["text", "intent", "features"])


def add_reduced_dimension(df):
    feature_mat = np.stack(df.loc[:, "features"].values)

    pca = PCA(n_components=2)
    pca.fit(feature_mat)
    X = pca.transform(feature_mat)

    df_pca = pd.DataFrame(X, columns=["x1", "x2"])

    return pd.concat([df, df_pca], axis=1)


def visualize_features(df: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(df).interactive()
    return (
        chart.mark_point()
        .encode(x="x1", y="x2", color="intent", tooltip="text")
        .properties(width=500, height=250)
    )


def get_intents(domain_path, n=None):
    intents = Domain.load(domain_path).intents

    if not n:
        return intents
    else:
        return np.random.choice(intents, n).tolist()
