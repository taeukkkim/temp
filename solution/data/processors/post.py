# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from transformers import EvalPrediction
from transformers.utils import logging
from solution.utils.constant import ANSWER_COLUMN_NAME


logger = logging.get_logger(__name__)


def save_pred_json(
    all_predictions, all_nbest_json, output_dir, prefix
):
    """
    Save prediction.json, nbest_predctions.json in output_dir.
    
    Args:   
        all_predictions ([Dict]): total prediction to be updated.
        all_nbest_json ([Dict]): total prediction of nbest size to be updated.
        output_dir ([str]): output directory.
        prefix ([str]): prefix to distinguish data to be stored.
    """
    
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

    prediction_file = os.path.join(
        output_dir,
        "predictions.json" if prefix is None else f"predictions_{prefix}.json",
    )
    nbest_file = os.path.join(
        output_dir,
        "nbest_predictions.json"
        if prefix is None
        else f"nbest_predictions_{prefix}.json",
    )

    logger.warning(f"Saving predictions to {prediction_file}.")
    with open(prediction_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
        )
    logger.warning(f"Saving nbest_preds to {nbest_file}.")
    with open(nbest_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
        )
        

def get_all_logits(
    predictions, features
):
    """After checking assertions against predictions and features length,
    return start & end logtis.

    Args:
        predictions ([Tuple[ndarray, ndarray]]): start & end logit predictions.
        features ([Dataset]): tokenized & splited datasets.

    Returns:
        Tuple([array]): start & end logtis
    """
    
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."
    
    return all_start_logits, all_end_logits


def map_features_to_example(
    examples, features
):
    """Returns the mapping of feature indices to example index.

    Args:
        examples ([Dataset]): raw datasets.
        features ([Dataset]): tokenized & splited datasets.

    Returns:
        [Dict[List]]: the mapping of feature indices to example index.
    """
    
    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    return features_per_example


def get_candidate_preds(
    features, feature_indices, all_start_logits, all_end_logits, n_best_size, max_answer_length
):
    """It returns predictions of n_best_size of features mapped to one exmaple.

    Args:
        features ([Dataset]): tokenized & splited datasets.
        feature_indices ([List]): feature indices of one loop exmaple.
        all_start_logits ([ndarray]): all start logits.
        all_end_logits ([ndarray]): all end logits.
        n_best_size ([int]): number of return best predictions.
        max_answer_length ([int]): max span of answer.

    Returns:
        [List[Dict]]: predictions of n_best_size of features mapped to one exmaple.
    """
    
    min_null_prediction = None
    prelim_predictions = []
    
    for feature_index in feature_indices:
        # 각 featureure에 대한 모든 prediction을 가져옵니다.
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        # logit과 original context의 logit을 mapping합니다.
        offset_mapping = features[feature_index]["offset_mapping"]
        # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
        token_is_max_context = features[feature_index].get(
            "token_is_max_context", None
        )

        # minimum null prediction을 업데이트 합니다.
        feature_null_score = start_logits[0] + end_logits[0]
        if (
            min_null_prediction is None
            or min_null_prediction["score"] > feature_null_score
        ):
            min_null_prediction = {
                "offsets": (0, 0),
                "score": feature_null_score,
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
            }

        # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
        start_indexes = np.argsort(start_logits)[
            -1 : -n_best_size - 1 : -1
        ].tolist()

        end_indexes = np.argsort(end_logits)[
            -1 : -n_best_size - 1 : -1
        ].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # out-of-scope answers는 고려하지 않습니다.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue
                # 최대 context가 없는 answer도 고려하지 않습니다.
                if (
                    token_is_max_context is not None
                    and not token_is_max_context.get(str(start_index), False)
                ):
                    continue
                prelim_predictions.append(
                    {
                        "offsets": (
                            offset_mapping[start_index][0],
                            offset_mapping[end_index][1],
                        ),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )

    # 가장 좋은 `n_best_size` predictions만 유지합니다.
    predictions = sorted(
        prelim_predictions, key=lambda x: x["score"], reverse=True
    )[:n_best_size]
    
    return predictions


def get_example_prediction(
    example, predictions, all_predictions, all_nbest_json
):  
    """Convert offset from a presentation from an excel to an answer text,
    and return by adding a presentation to all_prediction and all_nbest_json.

    Args:
        example ([Dataset]): raw datasets.
        predictions ([List[Dict]]): prediction of one example.
        all_predictions ([Dict]): total prediction to be updated.
        all_nbest_json ([Dict]): total prediction of nbest size to be updated.

    Returns:
        [List[Dict]]:
    """
    # predict text offset mapping
    context = example["context"]
    for pred in predictions:
        offsets = pred.pop("offsets")
        pred["text"] = context[offsets[0] : offsets[1]]

    # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
    if len(predictions) == 0 or (
        len(predictions) == 1 and predictions[0]["text"] == ""
    ):

        predictions.insert(
            0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
        )

    # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
    scores = np.array([pred.pop("score") for pred in predictions])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    # 예측값에 확률을 포함합니다.
    for prob, pred in zip(probs, predictions):
        pred["probability"] = prob

    # best prediction을 선택합니다.
    all_predictions[example["id"]] = predictions[0]["text"]
    
    # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
    all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]
    
    return all_predictions, all_nbest_json


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True, ##
):
    """
    A function that post-processes the presentation value of the QA model.
    since the model returns start logit and end logit, post-processing is required to change to original text based on this.

    Args:
        examples: raw dataset.
        features: preprocessed dataset
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            model predictions : start logits and two arrays representing the end logits,
            and the first dimension :obj:'features' element must match the number.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            total number of n-best presentations to generate when looking for answers
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            the maximum length of the answer that can be predicted.
        output_dir (:obj:`str`, `optional`):
            save path
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            prefix is included in the dictionary and stored.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            whether this process is the main process (used to determine whether logging/saving should be performed)
    
    Returns:
        [Dict]: total prediction of examples
    """
    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.warning(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
    
    # get logits with checking predictions and features
    all_start_logits, all_end_logits = get_all_logits(predictions, features)

    # map features to example
    features_per_example = map_features_to_example(examples, features)

    # 전체 example들에 대한 main Loop    
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = get_candidate_preds(
            features, feature_indices, all_start_logits, all_end_logits, n_best_size, max_answer_length
        )

        # offset을 활용해 text로 변환 후, all_prediction, all_nbest_json 업데이트
        all_predictions, all_nbest_json = get_example_prediction(
            example, predictions, all_predictions, all_nbest_json
        )
        
    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        save_pred_json(
            all_predictions, all_nbest_json, output_dir, prefix
        )

    return all_predictions


def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    log_level: Optional[int] = logging.WARNING,
):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 5:
        raise ValueError("`predictions` should be a tuple with five elements.")
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_start_top`/`n_end_top` greater start and end logits.
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    # Don't consider out-of-scope answers (last part of the test should be unnecessary because of the
                    # p_mask but let's not take any risk)
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length negative or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            predictions.insert(0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction and set the probability for the null answer.
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json


# Post-processing:
def post_processing_function(
    examples, 
    features, 
    predictions, 
    training_args, 
    mode,
):
    """
    main function of post_process

    Args:
        example ([Dataset]): raw datasets.
        features ([Dataset]): tokenized & splited datasets.
        predictions ([Dict]): predcitions
        training_args ([type]): arguments
        mode ([str]): status of model training(train, eval, predict)

    Returns:
        [Dict]: predictions
    """
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=training_args.max_answer_length,
        output_dir=training_args.output_dir,
        prefix=training_args.run_name + '_' + mode,
    )

    predictions, scores_diff_json = postprocess_qa_predictions_with_beam_search(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            start_n_top=model.config.start_n_top,
            end_n_top=model.config.end_n_top,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if mode == "predict":
        return formatted_predictions
    else:
        references = [
            {"id": ex["id"], "answers": ex[ANSWER_COLUMN_NAME]}
            for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )