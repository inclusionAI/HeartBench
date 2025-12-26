import time
from typing import Dict, List, Any

import math
import pandas as pd

from heartbench.utils import OpenAIService, parse_api_response

PROMPT_TEMPLATE = """你是一个专业、严苛的心理学打分批评专家，请根据以上对话历史（作为考题）与待测AI的表现进行打分。
注意事项：
1.考题中的角色B不是待测AI，仅辅助生成考题上文。仅「待测AI回复」中才是你要考察的部分。
2.打分要尽可能严格，加分项只有完全达到才能命中，扣分项如果有一点涉及都要命中。
3.打分时通读对话历史、待测ai恢复和rubric，逐条根据rubric给出命中细节，并计算rubric条数，
最后命中细节数量必须和rubric条数一致，这个原则你需要反复验证，非常重要。
4.不必在意分值多少，不需要计算总分。

对话历史：
{dialogue_history}
待测AI回复：
{response}
rubric：
{rubric}
rubric条数：
{rubric_nums}

直接按照格以下式输出，不要输出markdown或者其他内容。
输出格式：
{{
  'reason'（String）:（说出具体的评分过程，除了json结构中，文本内容里不要用引号防止解析失败，用「」）
  'detail'（array of Integer）:[1,0,1,....](按顺序给出rubric命中情况，命中置1，未命中置0，用英文逗号分割，这里的元素个数一定要与rubric条数保持一致)
}}"""


def calculate_dimension_details(
    rubric: List[Dict[str, Any]],
    detail: List[int],
    special_dimension: str = "其他"
) -> Dict[str, Any]:
    """
    Calculate per-dimension scores and question-level score for a single item.

    Per dimension D:
      - raw_score[D]: sum of scores of hit rubric items in D (can be negative)
      - min_score[D]: theoretical minimum total score in D on this question
                      (sum of all negative scores in this dimension)
      - max_score[D]: theoretical maximum total score in D on this question
                      (sum of all positive scores in this dimension)

    Normalized score for dimension D:
      norm_score[D] = log((raw_score[D] - min_score[D]) + 1) /
                      log((max_score[D] - min_score[D]) + 1)

    Special rules:
      - If any rubric of the special_dimension is hit (score > 0),
        then ALL dimension norm_scores and question_score = 0.
      - The special_dimension itself is NOT included in dimension_details,
        and is NOT used in question_score.
    """

    # 1. Guard: handle None or empty rubric
    if not rubric:
        return {
            "dimension_details": [],
            "question_score": 0.0,
            "has_special_hit": False
        }

    # 2. Align detail length with rubric length (pad or truncate)
    if len(detail) < len(rubric):
        detail = detail + [0] * (len(rubric) - len(detail))
    elif len(detail) > len(rubric):
        detail = detail[:len(rubric)]

    # 3. Build per-dimension min/max ranges from rubric
    dimension_ranges: Dict[str, Dict[str, float]] = {}
    for item in rubric:
        dim = item.get("dimension")
        score = item.get("score", 0)

        if dim is None:
            continue

        s = float(score) if score is not None else 0.0

        if dim not in dimension_ranges:
            dimension_ranges[dim] = {"min": 0.0, "max": 0.0}

        if s < 0:
            dimension_ranges[dim]["min"] += s
        elif s > 0:
            dimension_ranges[dim]["max"] += s

    # 4. Aggregate actual scores per dimension (only for hit items)
    raw_scores: Dict[str, float] = {dim: 0.0 for dim in dimension_ranges.keys()}
    has_special_hit = False

    for rub, hit in zip(rubric, detail):
        if not hit:
            continue

        dim = rub.get("dimension")
        score = rub.get("score", 0)

        if dim is None:
            continue

        s = float(score) if score is not None else 0.0
        raw_scores[dim] = raw_scores.get(dim, 0.0) + s

        # special dimension hit rule:
        if dim == special_dimension and s > 0:
            has_special_hit = True

    # 5. If no dimension got any score, everything is 0
    if not raw_scores:
        return {
            "dimension_details": [],
            "question_score": 0.0,
            "has_special_hit": has_special_hit
        }

    # 6. If special dimension is hit, all normalized scores and question score = 0
    #    and DO NOT output special_dimension in dimension_details.
    if has_special_hit:
        dimension_details = []
        for dim in dimension_ranges.keys():
            if dim == special_dimension:
                continue  # skip special dimension
            actual = raw_scores.get(dim, 0.0)
            dimension_details.append({
                "ability": dim,
                "raw_score": actual,
                "norm_score": 0.0
            })
        return {
            "dimension_details": dimension_details,
            "question_score": 0.0,
            "has_special_hit": True
        }

    # 7. Compute normalized score for each dimension separately
    dimension_details = []
    norms_for_avg = []

    for dim, range_info in dimension_ranges.items():
        if dim == special_dimension:
            # fully ignore special dimension in outputs and in averages
            continue

        min_score = range_info["min"]
        max_score = range_info["max"]
        actual_score = raw_scores.get(dim, 0.0)

        span = max_score - min_score
        if span <= 0:
            norm = 0.0
        else:
            numerator_base = (actual_score - min_score) + 1.0
            denominator_base = span + 1.0  # (max_score - min_score) + 1

            if numerator_base <= 0:
                numerator_base = 1.0
            if denominator_base <= 0:
                denominator_base = 1.0

            numerator = math.log(numerator_base)
            denominator = math.log(denominator_base)

            norm = numerator / denominator * 100 if denominator != 0 else 0.0

        norms_for_avg.append(norm)

        dimension_details.append({
            "ability": dim,
            "raw_score": actual_score,
            "norm_score": norm
        })

    # 8. Question score = average of norm scores of non-special dimensions
    question_score = sum(norms_for_avg) / len(norms_for_avg) if norms_for_avg else 0.0

    return {
        "dimension_details": dimension_details,
        "question_score": question_score,
        "has_special_hit": False
    }

class AnswerScorer:
    def __init__(self, config: Dict):
        """
        Initialize scorer
        :param config: Configuration dictionary containing api_key, model, base_url
        """
        self.openai_service = OpenAIService()
        self.openai_service.configure(config)

    def test_connection(self):
        """Test API connection"""
        print("Testing API connection...")
        self.openai_service.test_connection()

    def score_answers(self,
                      input_file: str,
                      max_workers: int = 2,
                      max_retries: int = 3
                      ) -> pd.DataFrame:
        """
        Score answers
        :param input_file: Excel file containing answers
        :param max_workers: Number of concurrent requests
        :param max_retries: Maximum retry attempts
        :return: DataFrame containing scores
        """
        # 1. Read file
        print(f"Reading file: {input_file}")
        df = pd.read_json(input_file, lines=True)
        print(f"Total {len(df)} records read")

        # Check required columns
        required_cols = ['question_id', 'dialogue', 'response', 'rubric']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"File missing required columns: {missing_cols}")

        # 2. Prepare batch requests
        print("\nPreparing batch requests...")
        messages_list = []
        valid_indices = []

        for idx, row in df.iterrows():
            # Skip rows with empty or error answers
            if pd.isna(row['response']) or (isinstance(row['response'], str) and row['response'].startswith("Error")):
                print(f"Skipping question_id {row['question_id']}: answer is empty or error")
                continue

            dialogue = row['dialogue']
            response = row['response']
            rubric = row['rubric']
            rubric_str = "\n".join(
                f"[{item['dimension']}][{item['score']}] {item['content']}"
                for item in rubric
            )
            rubric_nums = len(rubric)

            prompt = PROMPT_TEMPLATE.format(dialogue_history=dialogue, response=response, rubric=rubric_str,
                                            rubric_nums=rubric_nums)
            messages_list.append([{"role": "user", "content": prompt}])
            valid_indices.append(idx)

        print(f"Prepared {len(messages_list)} valid requests")

        if len(messages_list) == 0:
            print("⚠ Warning: No valid data to score")
            return df

        # 3. Batch API calls (with retry)
        print(f"\n{'=' * 80}")
        print(f"Starting batch scoring...")
        print(f"{'=' * 80}")

        results = self._process_batch_with_retry(
            messages_list, max_workers, max_retries
        )

        # 5. Parse results and calculate scores
        print(f"\n{'=' * 80}")
        print(f"Processing results and calculating scores...")
        print(f"{'=' * 80}")

        # Initialize new columns
        df['score_detail'] = None
        df['score_reason'] = None
        df['dimension_score'] = None

        success_count = 0
        error_count = 0

        special_dim = "其他"

        for i, idx in enumerate(valid_indices):
            row = df.loc[idx]
            rubric = row['rubric']

            api_response = results[i]

            if isinstance(api_response, str) and api_response.startswith("Error"):
                print(f"✗ Question {row['question_id']} API call failed")
                df.at[idx, 'api_response'] = api_response
                error_count += 1
            else:
                # Parse response
                detail, reason = parse_api_response(api_response)

                if len(detail) == 0:
                    print(f"✗ Question {row['question_id']} parse failed")
                    df.at[idx, 'api_response'] = api_response
                    df.at[idx, 'score_reason'] = reason
                    error_count += 1
                else:

                    # Calculate dimension score details
                    dim_results = calculate_dimension_details(rubric, detail, special_dim)

                    df.at[idx, 'score_detail'] = str(detail)
                    df.at[idx, 'score_reason'] = reason
                    df.at[idx, "dimension_score"] = dim_results["dimension_details"]
                    df.at[idx, "question_score"] = dim_results["question_score"]

                    success_count += 1
                    print(f"✓ Question {row['question_id']}: "
                          f"question_score={dim_results['question_score']}")

        print(f"\n{'=' * 80}")
        print(f"Scoring completed!")
        print(f"{'=' * 80}")
        print(f"✓ Success: {success_count}")
        print(f"✗ Failed: {error_count}")

        # 6. Calculate total score and dimension scores
        print("\nCalculating total score and dimension scores...")
        dimension_score_df = self._calculate_scores(df)

        # 8. Save results
        print(f"\nSaving results to: question_score.jsonl and overall_score_results.jsonl")

        # 8.1 save per-question scores
        question_output_file = "./data/question_score.jsonl"
        df.to_json(question_output_file, orient="records", lines=True, force_ascii=False)

        # 8.2 save per-dimension aggregated scores + one extra row for overall score
        overall_output_file = "./data/overall_score_results.jsonl"

        # append a row for overall score
        overall_row = {
            "ability": "OVERALL",
            "dimension_score": self.overall_score,
            "question_count": None
        }
        dimension_score_with_overall = pd.concat(
            [dimension_score_df, pd.DataFrame([overall_row])],
            ignore_index=True
        )

        dimension_score_with_overall.drop(columns=["question_count"], inplace=True, errors="ignore")

        dimension_score_with_overall.to_json(
            overall_output_file,
            orient="records",
            lines=True,
            force_ascii=False
        )

        print(f"Question-level results saved to: {question_output_file}")
        print(f"Dimension-level results saved to: {overall_output_file}")
        # print(f"Overall score (average over all dimensions): {self.overall_score:.4f}")

        return df

    def _calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate dimension scores across all questions.

        Assumes df already contains:
          - 'dimension_score': per-question dimension details, a list of dicts:
              [
                {"ability": <dimension_name>, "raw_score": <int>, "norm_score": <float>},
                ...
              ]
          - 'question_score': per-question total score (float)

        Returns:
          - dimension_score_df: per-dimension aggregated scores with columns:
              ['ability', 'dimension_score', 'question_count']
          And sets:
          - self.overall_score: average of all dimension scores.
        """

        print("\nAggregating dimension scores over all questions...")

        sum_norm: Dict[str, float] = {}
        count: Dict[str, int] = {}

        for _, row in df.iterrows():
            dim_details = row.get("dimension_score")
            if not dim_details:
                continue

            # dim_details: list of {"ability", "raw_score", "norm_score"}
            for d in dim_details:
                dim_name = d.get("ability")
                if dim_name is None:
                    continue

                norm_score = d.get("norm_score", 0.0)
                norm_score = float(norm_score) if norm_score is not None else 0.0

                sum_norm[dim_name] = sum_norm.get(dim_name, 0.0) + norm_score
                count[dim_name] = count.get(dim_name, 0) + 1

        records = []
        for dim_name in sorted(sum_norm.keys()):
            c = count.get(dim_name, 0)
            avg_norm = sum_norm[dim_name] / c if c > 0 else 0.0

            records.append({
                "ability": dim_name,
                "dimension_score": avg_norm,
                "question_count": c,
            })

        dimension_score_df = pd.DataFrame(records)

        if not dimension_score_df.empty:
            overall_score = float(dimension_score_df["dimension_score"].mean())
        else:
            overall_score = 0.0

        self.overall_score = overall_score
        print(f"\nOverall score (average over all dimensions): {overall_score:.4f}")

        return dimension_score_df

    def _process_batch_with_retry(self, messages_list: List,
                                  max_workers: int,
                                  max_retries: int) -> List[str]:
        """Batch processing with retry mechanism"""
        results = self.openai_service.get_completion_batch(
            messages_list=messages_list,
            max_workers=max_workers,
            desc="🎯 Batch scoring"
        )

        # Check failures and retry
        retry_indices = []
        retry_messages = []

        for idx, result in enumerate(results):
            if isinstance(result, str) and result.startswith("Error"):
                retry_indices.append(idx)
                retry_messages.append(messages_list[idx])

        retry_count = 0
        while retry_messages and retry_count < max_retries:
            retry_count += 1
            print(f"\nRetry attempt {retry_count}, {len(retry_messages)} failed requests...")
            time.sleep(2)

            retry_results = self.openai_service.get_completion_batch(
                messages_list=retry_messages,
                max_workers=max_workers,
                desc=f"🔄 Retry ({retry_count}/{max_retries})"
            )

            new_retry_indices = []
            new_retry_messages = []

            for i, idx in enumerate(retry_indices):
                if not (isinstance(retry_results[i], str) and retry_results[i].startswith("Error")):
                    results[idx] = retry_results[i]
                else:
                    new_retry_indices.append(idx)
                    new_retry_messages.append(retry_messages[i])

            retry_indices = new_retry_indices
            retry_messages = new_retry_messages

        if retry_messages:
            print(f"\nWarning: {len(retry_messages)} requests still failed")

        return results
