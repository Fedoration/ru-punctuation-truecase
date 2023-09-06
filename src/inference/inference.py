from config import MODEL_MAX_LENGTH, MODEL_NAME, PATH_TO_CHECKPOINT
from model import ReCapitalizationModel


def main():
    # Пример инференса
    recapitalization_model = ReCapitalizationModel(
        path_to_checkpoint=PATH_TO_CHECKPOINT,
        model_name=MODEL_NAME,
        model_max_length=MODEL_MAX_LENGTH,
        is_question=True,
    )

    test_queries = ["кто такая алла пугачёва", "столица россии", "что такое москва"]

    recapitalized_queries = recapitalization_model.restore_capitalization(test_queries)
    for query, result in zip(test_queries, recapitalized_queries):
        print(f"Query   : {query}")
        print(f"Recapitalized: {result.strip()}\n")


if __name__ == "__main__":
    main()
