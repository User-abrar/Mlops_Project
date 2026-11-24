import argparse
import subprocess
import sys

from prefect import flow, task
from prefect.cache_policies import NONE

from main import etape_prepare
from model_pipeline import train_model, evaluate_model, save_model, load_model

PY = sys.executable
FILES = "model_pipeline.py main.py pipeline_prefect.py"
DATA = "diabetic_data.csv"
MODEL = "model.joblib"
N_ITER = 15


def run(cmd):
    subprocess.run(cmd.split(), check=True)


@task
def install_deps():
    run(f"{PY} -m pip install -r requirements.txt")


@task
def format_code():
    run(f"{PY} -m black {FILES}")


@task
def quality():
    run(f"{PY} -m flake8 {FILES} --max-line-length=100")


@task
def security():
    run(f"{PY} -m bandit -ll {FILES}")


@task
def tests():
    run(f"{PY} -m pytest -v")


@task(cache_policy=NONE)
def prepare():
    return etape_prepare(DATA)


@task(cache_policy=NONE)
def train(X_train, y_train):
    return train_model(X_train, y_train, n_iter=N_ITER)


@task(cache_policy=NONE)
def save(model):
    save_model(model, MODEL)


@task(cache_policy=NONE)
def load():
    return load_model(MODEL)


@task(cache_policy=NONE)
def evaluate(model, X_test, y_test, noms_classes):
    evaluate_model(model, X_test, y_test, noms_classes)


@flow(name="code")
def code_flow():
    install_deps()
    format_code()
    quality()
    security()
    tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test, noms = prepare()
    train(X_train, y_train)


@flow(name="evaluate")
def evaluate_flow():
    X_train, X_test, y_train, y_test, noms = prepare()
    model = load()
    evaluate(model, X_test, y_test, noms)


@flow(name="all")
def all_flow():
    code_flow()
    X_train, X_test, y_train, y_test, noms = prepare()
    model = train(X_train, y_train)
    save(model)
    evaluate(model, X_test, y_test, noms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow", default="all", choices=["all", "train", "evaluate", "code"]
    )
    args = parser.parse_args()
    flows = {
        "all": all_flow,
        "train": train_flow,
        "evaluate": evaluate_flow,
        "code": code_flow,
    }
    flows[args.flow]()
