import pandas as pd

from typing import Dict

from heartbench.utils import OpenAIService

PROMPT_TEMPLATE = """你作为一个人类的真诚伙伴，正在与人类对话。
对话历史如下：
{dialogue_content}
请将以上对话历史视为真实的交流，给出你的下文回应。不要使用Markdown格式。"""


class AnswerGenerator:
    def __init__(self, config: Dict):
        """
        Initialize answer generator
        :param config: Configuration dictionary containing api_key, model, base_url
        """
        self.openai_service = OpenAIService()
        self.openai_service.configure(config)

    def test_connection(self):
        """Test API connection"""
        print("Testing API connection...")
        self.openai_service.test_connection()

    def generate_answers(self,
                         input_file: str,
                         output_file: str,
                         mode: str = 'all',
                         max_workers: int = 2
                         ) -> pd.DataFrame:
        """
        Generate answers
        :param input_file: Input Excel file path
        :param output_file: Output Excel file path
        :param mode: Question mode - 'all', 'hard', 'normal'
        :param max_workers: Number of concurrent requests
        :return: DataFrame containing answers
        """
        # 1. Read JSONL file
        print(f"\nReading JSONL file: {input_file}")
        df = pd.read_json(input_file, lines=True)
        print(f"Total rows read: {len(df)}")

        # 2. Filter by mode
        if mode == 'hard':
            df = df[df['difficulty'] == 'hard'].copy()
            print(f"Filtered hard questions, {len(df)} remaining")
        elif mode == 'normal':
            df = df[df['difficulty'] == 'normal'].copy()
            print(f"Filtered normal questions, {len(df)} remaining")
        else:
            df = df.copy()
            print(f"Processing all questions")

        # 3. Prepare batch requests
        print("\nPreparing batch requests...")
        messages_list = []
        for idx, row in df.iterrows():
            dialogue = row['dialogue']
            prompt = PROMPT_TEMPLATE.format(dialogue_content=dialogue)
            messages_list.append([{"role": "user", "content": prompt}])

        # 4. Batch API calls
        print(f"\n{'=' * 80}")
        print(f"Starting answer generation (mode: {mode}, count: {len(messages_list)})...")
        print(f"{'=' * 80}")

        results = self.openai_service.get_completion_batch(
            messages_list=messages_list,
            max_workers=max_workers,
            desc=f"🚀 Generating answers"
        )

        # 5. Add responses to DataFrame
        df['response'] = results

        # 6. Count success and failures
        success_count = sum(1 for r in results if not (isinstance(r, str) and r.startswith("Error")))
        error_count = len(results) - success_count

        print(f"\n{'=' * 80}")
        print(f"Generation completed!")
        print(f"{'=' * 80}")
        print(f"✓ Successfully generated: {success_count}")
        print(f"✗ Failed: {error_count}")

        # 7. Save results
        print(f"\nSaving results to: {output_file}")
        df.to_json(output_file, orient="records", lines=True, force_ascii=False)

        return df
