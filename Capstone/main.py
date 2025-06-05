from preprocess import preprocess
from model import train_model


def main():
    # TODO: I want to tweak the noise variables, to see how they affect the model.
    #       This way we can see how the model performs with different levels of noise.
    #       Currently, the noise variables are hardcoded in the preprocess.py file.
    #       And the model accuracy is currently 0.95, which is too high.
    #       I want to see how the model performs with different levels of noise.
    preprocess()
    model_info = train_model()
    print("Model training complete.")
    print("Best Parameters:", model_info["grid_search"].best_params_)
    print("Accuracy:", model_info["grid_search"].best_score_)
    print("\nClassification Report:\n", model_info["grid_search"].best_estimator_.classification_report(model_info["y_test"], model_info["grid_search"].best_estimator_.predict(model_info["X_test_selected"])))
    print("\nTop features:")
    print(model_info["grid_search"].best_estimator_.feature_importances_[:10])

if __name__ == "__main__":
    main()