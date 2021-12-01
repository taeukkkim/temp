from .odqa import OdqaProcessor, convert_examples_to_features
from .post import post_processing_function


POST_PROCESSING_FUNCTION = {
    "extractive": post_processing_function,
}
