# Model
MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
SHORT_MODEL_NAME = MODEL_NAME.split("/")[1] if "/" in MODEL_NAME else MODEL_NAME
MODEL_MAX_LENGTH = 512

# Data
PATH_TO_DATA = "../data"
