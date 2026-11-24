import subprocess
import sys

import model_pipeline


def test_functions_exist():
    assert callable(model_pipeline.prepare_data)
    assert callable(model_pipeline.train_model)
    assert callable(model_pipeline.evaluate_model)


def test_save_load_exist():
    assert callable(model_pipeline.save_model)
    assert callable(model_pipeline.load_model)


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "test_model.joblib")
    model_pipeline.save_model({"a": 1}, path)
    assert model_pipeline.load_model(path) == {"a": 1}


def test_prepare_data_output():
    result = model_pipeline.prepare_data("diabetic_data.csv")
    assert len(result) == 5
    X_train, X_test, y_train, y_test, noms = result
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert len(noms) == 3


def test_main_help():
    r = subprocess.run([sys.executable, "main.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--step" in r.stdout
