<div align="center">
  <h1 style="margin:16px 0 8px;">HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs</h1>
</div>
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0">
</a>

---

## 🎯 Introduction

HeartBench is an evaluation benchmark for the psychological and social sciences field, designed to transcend traditional
knowledge and reasoning assessments. It focuses on measuring large language models' (LLMs) anthropomorphic capabilities
in human-computer interactions, covering dimensions such as personality, emotion, social skills, and ethics.

- Evaluation Samples: 296 multi-turn dialogues
- Scoring Criteria (Rubric): 2,818 items
- Scenarios: 33 scenarios (e.g., personal growth, family relationships, workplace psychology)
- Evaluation Dimensions: 5 anthropomorphism capability categories and 15 specific anthropomorphic abilities (e.g. curiosity, warmth, emotional understanding)
Learn more in our research paper.

## 💡 Key Features

1. **Real-World Alignment:** Our dataset is built from anonymized and rewritten dialogues between real users and counselors, covering high-frequency scenarios like family relationships, personal growth, and workplace psychology. We move beyond simple fact-based Q&A by employing multi-turn dialogue evaluation. The focus is on assessing a model's ability to understand complex emotions and respond to social contexts within long conversations and their subtext, rather than its capacity for simple mimicry.
2. **Fine-Grained, Science-Based Evaluation:** We have developed the "AI Human-like Capability Framework," a sophisticated evaluation system rooted in established psychological theories. This framework assesses models across 5 core capabilities and 15 fine-grained subcategories, including personality traits, emotional intelligence, and social skills. For each dialogue, our expert team has authored between 4 and 15 specific scoring criteria.
3. **Co-developed with Domain Experts:** The benchmark was created in close collaboration with experts in psychology and anthropology. Their involvement spanned the entire process: from the construction of the corpus using authentic counseling data, to the identification of over 200 key evaluation points, and the formulation of more than 3,000 scientific scoring rubrics. All data was then rigorously annotated and reviewed by these experts to ensure quality and accuracy.

## 🏆 Benchmark Results

We evaluated the performance of current leading models on HeartBench, scoring their performance in each dimension on a scale of 0 to 100. The table below shows the overall results for each model across all test samples.

### Main Results

| Model | Score |
|-------|-------|
| Claude-sonnet-4.5-20250929 | 62.65 |
| gemini-3-pro-preview | 61.54 |
| Qwen3-235B-A22B-instruct-2507 | 61.47 |
| Qwen3-next-80B-A3B-Instruct | 61.09 |
| Qwen3-30B-A3B-instruct-2507 | 60.16 |
| gpt-5-2025-08-07 | 60.16 |
| Gemini-2.5-pro | 59.85 |
| Ling-1T | 59.82 |
| KIMI-K2-Instruct-0905 | 57.97 |
| gpt-4.1-2025-04-14 | 51.62 |
| Qwen3-30B-A3B | 48.21 |
| gpt-4o-2024-11-20 | 48.20 |
| DeepSeek-V3.2-Exp | 47.43 |

### Results Across 15 Abilities

![](https://oss-ata.alibaba.com/article/2025/12/d94e952a-1340-4ab6-b814-8b58107595b2.png)

## 📊 Dataset

### Evaluation Dimensions

HeartBench is built upon the psychological theory of "Anthropomorphic Intelligence" Drawing inspiration from psychology's classification of human mental functions, it evaluates models across 5 core anthropomorphic ability categories and 15 specific ability.

🧠 **Personality:** Ability to project an independent, autonomous, and agreeable persona. This is demonstrated through a natural language style, a sense of humor, autonomy, other positive human-like traits, and stable self-esteem and self-awareness.  

😊 **Emotion:** Ability to exhibit appropriate emotional responses and to effectively perceive, understand, and respond to the emotional states of others.

🤝 **Social:** Ability to demonstrate a strong willingness for social interaction and to effectively build interpersonal relationships.

⚖️ **Morality:** Ability to operate based on the moral norms and ethical principles of human society. This includes acutely identifying moral dilemmas within a situation, expressing an understanding of these issues, and providing morally sound decisions or advice.

🎯 **Motivation:** Ability to articulate rational, clear, and self-consistent motivations for its own statements and actions, while also being able to understand and reasonably infer the underlying motivations of others based on contextual clues.

| Ability              | Rubric Count (%) |
| :------------------- |:-----------------|
| **Personality**      | **1634 (39%)**   |
| &nbsp;&nbsp;Verbal Expression | 565 (20.0%)      |
| &nbsp;&nbsp;Curiosity         | 367 (13.0%)      |
| &nbsp;&nbsp;Warmth            | 305 (10.8%)      |
| &nbsp;&nbsp;First-Person Usage| 295 (10.5%)      |
| &nbsp;&nbsp;Autonomy          | 37 (1.3%)        |
| &nbsp;&nbsp;Humor             | 36 (1.3%)        |
| &nbsp;&nbsp;Self-Awareness    | 29 (1.0%)        |
|                       |                  |
| **Emotion**          | **1015 (36%)**   |
| &nbsp;&nbsp;Emotional Coping     | 390 (13.8%)      |
| &nbsp;&nbsp;Emotional Understanding | 309 (11.0%)      |
| &nbsp;&nbsp;Emotional Perception  | 284 (10.1%)      |
| &nbsp;&nbsp;Emotional Reaction    | 32 (1.1%)        |
|                       |                  |
| **Social**           | **104 (3.7%)**   |
| &nbsp;&nbsp;Proactivity         | 79 (2.8%)        |
| &nbsp;&nbsp;Relationship Building | 25 (0.9%)        |
|                       |                  |
| **Motivation**       | **42 (1.5%)**    |
|                       |                  |
| **Morality**         | **23 (0.8%)**    |
|                       |                  |
| **Total**            | **2818 (100%)**  |

### Scenario Distribution

Our dataset, `data/question_all.jsonl`, contains 296 meticulously designed multi-turn dialogues covering 33 real-world scenarios:

| Dialogue Scenario                 | Count (%) |
| :-------------------------------- | :-------- |
| Personal Growth                   | 110 (37.2%) |
| Interpersonal & Social Development | 66 (22.3%)  |
| Workplace Psychology              | 53 (17.9%)  |
| Family Relationships              | 37 (12.5%)  |
| Intimate Relationships            | 30 (10.1%)  |
| **Total**                         | **296 (100%)** |


### Data Sample

Each evaluation sample includes:
- **Context:** The multi-turn conversation history between users.
- **Question:** The final user utterance in the conversation. This serves as the prompt for the model to respond to and contains the specific points for evaluation.
- **Rubrics:** A set of high-quality scoring criteria, each detailing the evaluation dimension, score, and specific grading rules.

![img](https://oss-ata.alibaba.com/article/2025/12/815c02c7-729a-4d37-9880-ac875de7d538.png)

### Evaluation Method

We use the **"LLM-as-a-Judge"** method for objective, scalable evaluation of Anthropomorphic Intelligence qualities.
- **Judge:** **Claude 4.5 Sonnet** is our default judge, chosen for its nuanced understanding.
- **Process:** The judge views the full conversation and responses from multiple models. It then scores each response against a set of rubrics, providing a detailed rationale.
- **Validation:** We confirmed our method's reliability with an expert blind test. A review of 30% of the samples by 20+ psychology professionals showed an **86% human-LLM agreement rate** when scoring 14 top models.

## 🚀 Quick Start

### Prepare

```bash
pip install -r requirements.txt
```

You need to prepare an **OpenAIService** **API_KEY** and **BASE_URL** that can access the **claude-sonnet-4-5-20250929** model for model assessment.

### Usage

**Run all questions**

```
python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --mode all --model Model
```

**Score only (for assess your own model response)**
 > If you want to evaluate responses already generated by your own model, you need to prepare a jsonl file that contains all the questions from the data folder’s question jsonl file. For each entry, add your model’s answer in a **response** field corresponding to the same question_id. 
 >



```
python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --score_only --answer_file ./your_model_answers.jsonl
```



### Run a Example Evaluation for claude-sonnet-4-5

> This is an example evaluation script that can assess **claude-sonnet-4-5** in **all** question modes, including answer generation and assessment. 
>



```bash
export API_KEY=xxxx
export BASE_URL=xxxx

bash example.sh
```



---

## ⚖️ Ethics & Use

1. **For academic research and model evaluation purposes exclusively.** The use of this benchmark is strictly forbidden for replacing professional psychological counseling, making clinical diagnoses, or for the development of any form of automated therapeutic applications.

2. To safeguard privacy and mitigate risks, potentially sensitive or high-risk portions of the data have undergone **anonymization**. We advocate that users remain highly attentive to the **ethical boundaries and societal implications** of model outputs and interpret performance on complex tasks with due diligence.

3. When this data is used in any context that may involve real individuals (such as in clinical studies), it is **mandatory to ensure the supervision and guidance of a certified professional**. All activities must also strictly adhere to applicable local laws, regulations, and ethical guidelines.

   

### Copyright

- Author: Ant-DILab, Beijing Normal University

### Citation

```bibtex
@misc{liu2025heartbenchprobingcoredimensions,
      title={HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs}, 
      author={Jiaxin Liu and Peiyi Tu and Wenyu Chen and Yihong Zhuang and Xinxia Ling and Anji Zhou and Chenxi Wang and Zhuo Han and Zhengkai Yang and Junbo Zhao and Zenan Huang and Yuanyuan Wang},
      year={2025},
      eprint={2512.21849},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.21849}
