import argparse
from heartbench.generate_answers import AnswerGenerator
from heartbench.score_answers import AnswerScorer
import os


def main():
    """Main control script that chains answer generation and scoring steps"""
    parser = argparse.ArgumentParser(
        description='Dialogue Evaluation System - Generate Answers and Score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example usage:
      # Run all questions
      python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --mode all --model Model
    
      # Run only hard questions
      python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --mode hard --model Model
    
      # Run only normal questions
      python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --mode normal --model Model

    
      # Generate answers only, no scoring
      python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --mode all --generate_only
    
      # Score only (requires existing answer file)
      python run_evaluation.py --base_url YOUR_URL --api_key YOUR_KEY --score_only --answer_file ./data/temp_answers.jsonl
        """
    )

    # Basic parameters
    parser.add_argument('--input_file', type=str,
                        default='./data/question_all.jsonl',
                        help='Input jsonl file path')

    parser.add_argument('--mode', type=str,
                        default='all',
                        choices=['all', 'hard', 'normal'],
                        help='Question mode: all, hard, normal')

    # API configuration
    parser.add_argument('--api_key', type=str, required=True,
                        help='API key (required)')
    parser.add_argument('--model', type=str,
                        help='Model to use')
    parser.add_argument('--base_url', type=str,
                        help='API base URL (required)')

    # Execution control
    parser.add_argument('--generate_only', action='store_true',
                        help='Only generate answers, no scoring')
    parser.add_argument('--score_only', action='store_true',
                        help='Only score, no generation (requires existing answer file)')
    parser.add_argument('--answer_file', type=str, default=None,
                        help='When using --score_only, specify the file path containing answers')

    # Performance parameters
    parser.add_argument('--max_workers', type=int, default=2,
                        help='Number of concurrent requests')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum retry attempts for scoring')

    args = parser.parse_args()

    # Parameter validation
    if args.score_only and not args.answer_file:
        parser.error("Must specify --answer_file when using --score_only")

    if args.generate_only and args.score_only:
        parser.error("Cannot use --generate_only and --score_only simultaneously")

    # Configure API for answer generation
    generate_config = {
        "api_key": args.api_key,
        "model": args.model,
        "base_url": args.base_url,
    }

    # Configure API for scoring (use generation config if not specified)
    score_config = {
        "api_key": args.api_key,
        "model": "claude-sonnet-4-5-20250929",
        "base_url": args.base_url,
    }

    # Intermediate file path
    answer_file = args.answer_file or './data/temp_answers.jsonl'

    try:
        # Step 1: Generate answers
        if not args.score_only:
            print(f"\n{'=' * 80}")
            print(f"Step 1/2: Generate Answers")
            print(f"{'=' * 80}\n")

            generator = AnswerGenerator(generate_config)
            generator.test_connection()

            generator.generate_answers(
                input_file=args.input_file,
                output_file=answer_file,
                mode=args.mode,
                max_workers=args.max_workers,
            )

            print(f"\n✓ Answers saved to: {answer_file}")

            if args.generate_only:
                print(f"\n{'=' * 80}")
                print(f"✓ Completed! (Generate-only mode)")
                print(f"{'=' * 80}")
                return

        # Step 2: Score answers
        print(f"\n{'=' * 80}")
        print(f"Step {2 if not args.score_only else '1/1'}: Score Answers")
        print(f"{'=' * 80}\n")

        scorer = AnswerScorer(score_config)
        scorer.test_connection()

        scorer.score_answers(
            input_file=answer_file,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )

        print(f"\n{'=' * 80}")
        print(f"✓✓✓ All Completed! ✓✓✓")
        print(f"{'=' * 80}")

    except KeyboardInterrupt:
        print("\n\n⚠ User interrupted execution")
    except Exception as e:
        print(f"\n\n✗ Execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
