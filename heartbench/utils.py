import ast
import concurrent.futures
import json
import re
import threading
import time
from typing import List, Dict, Union, Optional
from typing import Tuple

import openai
from tqdm import tqdm


class OpenAIService:
    def __init__(self):
        self.client = None
        self.is_configured = False
        self.config = {
            'api_key': '',
            'base_url': '',
            'model': '',
            'max_tokens': 10000
        }
        # Add rate limiter (QPS=5)
        self.rate_limiter = RateLimiter(5)
        # Add thread-safe counter
        self.request_counter = 0
        self.counter_lock = threading.Lock()

    def configure(self, config: dict) -> None:
        """Configure OpenAI client with parameter validation and error handling best practices"""
        try:
            # Validate required parameters
            if not config.get('api_key') or not config['api_key'].strip():
                raise ValueError('API Key is required')

            # Format base URL
            base_url = config.get('base_url') or self.config['base_url']
            if not base_url.endswith('/v1'):
                base_url = re.sub(r'/$', '', base_url) + '/v1'

            # Validate model name
            model = config.get('model') or self.config['model']
            if not model or not model.strip():
                raise ValueError('Model name is required')

            # Create OpenAI client instance
            self.client = openai.OpenAI(
                api_key=config['api_key'].strip(),
                base_url=base_url
            )

            # Update configuration
            self.config = {
                'api_key': config['api_key'].strip(),
                'base_url': base_url,
                'model': model.strip(),
            }
            self.is_configured = True

        except Exception as e:
            print(f'❌ Configuration failed: {str(e)}')
            self.is_configured = False
            raise

    def _format_messages(self, messages: Union[str, List[Dict]]) -> List[Dict]:
        """Format messages into required API structure"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def get_completion(self,
                       messages: Union[str, List[Dict]],
                       options: Optional[dict] = None) -> str:
        """
        Synchronously get response with detailed error handling
        """
        if not self.is_configured:
            raise RuntimeError('Client not configured. Call configure() first.')

        formatted_messages = self._format_messages(messages)
        params = {
            "model": self.config['model'],
            "messages": formatted_messages,
            **(options or {})
        }

        try:
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content
        except Exception as e:
            self._handle_api_error(e)
            raise

    def _handle_api_error(self, error: Exception) -> None:
        """Unified API error handling"""
        error_map = {
            401: 'Invalid API Key, please check your key',
            403: 'API access denied, check permissions',
            404: 'API endpoint not found, check base_url',
            429: 'Rate limit exceeded, please retry later',
        }

        status = getattr(error, 'status_code', None)
        if status in error_map:
            err_msg = f"{error_map[status]} (HTTP {status})"
        elif 'ENOTFOUND' in str(error) or 'ECONNREFUSED' in str(error):
            err_msg = "Network connection failed, check base_url and network"
        else:
            err_msg = f"API request failed: {str(error)}"

        print(f'❌ API Error: {err_msg}')
        raise RuntimeError(err_msg)

    def test_connection(self) -> bool:
        """Test connection with diagnostic information"""
        if not self.is_configured:
            raise RuntimeError('Client not configured')

        try:
            self.get_completion("Hi", {"max_tokens": 5})
            return True
        except Exception as e:
            print(f'❌ Connection failed: {str(e)}')
            if 'model' in str(e):
                print('💡 Hint: Please confirm the model name is correct')
            raise

    def get_config(self) -> dict:
        return self.config.copy()

    def update_config(self, updates: dict) -> None:
        self.config.update(updates)
        if self.is_configured:
            self.configure(self.config)

    def is_ready(self) -> bool:
        return self.is_configured

    def get_completion_batch(
            self,
            messages_list: List[Union[str, List[Dict]]],
            options: Optional[dict] = None,
            max_workers: int = 2,
            show_progress: bool = True,
            desc: str = "Processing requests"
    ) -> List[str]:
        """
        Batch get API responses (multi-threaded + QPS limit + progress bar)
        :param messages_list: List of messages (each element represents an independent request)
        :param options: Request options
        :param max_workers: Maximum number of threads (default 5)
        :param show_progress: Whether to show progress bar (default True)
        :param desc: Progress bar description text
        :return: List of response contents (order matches input)
        """
        if not self.is_configured:
            raise RuntimeError('Client not configured. Call configure() first.')

        total_tasks = len(messages_list)
        results = [None] * total_tasks

        # Use ThreadPoolExecutor to manage thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to thread pool
            future_to_index = {
                executor.submit(
                    self._thread_safe_completion,
                    messages,
                    options,
                    idx
                ): idx for idx, messages in enumerate(messages_list)
            }

            # Create progress bar
            with tqdm(total=total_tasks, desc=desc, disable=not show_progress) as pbar:
                # Collect results (maintain original order)
                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"Error: {str(e)}"

                    # Update progress bar
                    pbar.update(1)

        return results

    def _thread_safe_completion(
            self,
            messages: Union[str, List[Dict]],
            options: Optional[dict],
            task_id: int
    ) -> str:
        """
        Thread-safe request wrapper (with QPS control)
        """
        # Acquire rate limit permission (blocks until token obtained)
        self.rate_limiter.acquire()

        # Update request counter (thread-safe)
        with self.counter_lock:
            self.request_counter += 1

        # Call original synchronous request method
        return self.get_completion(messages, options)


class RateLimiter:
    """QPS limiter (token bucket algorithm implementation)"""

    def __init__(self, rate: int):
        """
        :param rate: Number of requests allowed per second (QPS)
        """
        self.rate = rate
        self.tokens = rate
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            # Calculate time difference and refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            if elapsed > 1:  # Reset if more than 1 second
                self.tokens = self.rate
                self.last_refill = now
            elif elapsed > 0:
                # Refill tokens proportionally by time
                self.tokens = min(
                    self.rate,
                    self.tokens + elapsed * self.rate
                )
                self.last_refill = now

            # Check if tokens are sufficient
            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Calculate wait time needed
            wait_time = (1 - self.tokens) / self.rate
            self.last_refill = now + wait_time

        # Wait outside lock (avoid blocking other threads)
        time.sleep(wait_time)
        return self.acquire()  # Recursive retry


def parse_api_response(response_text: str, debug: bool = False) -> Tuple[List[int], str]:
    """解析API响应"""
    if debug:
        print("\n" + "=" * 80)
        print("开始解析API响应")
        print("=" * 80)
        print("原始响应内容:")
        print(response_text[:1000])
        print("-" * 80)

    try:
        response_text = response_text.strip()
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*\n', '', response_text)
            response_text = re.sub(r'\n```\s*$', '', response_text)

        # ================= 方法1: ast.literal_eval =================
        try:
            if debug:
                print("尝试方法1: ast.literal_eval")
            response_dict = ast.literal_eval(response_text)
            detail = response_dict.get('detail', [])
            reason = response_dict.get('reason', '')
            if debug:
                print(f"✓ 方法1成功! detail长度: {len(detail)}")
            return detail, reason
        except Exception as e:
            if debug:
                print(f"✗ 方法1失败: {e}")

        # ================= 方法2: 替换单引号为双引号 + json =================
        try:
            if debug:
                print("\n尝试方法2: 替换单引号为双引号")
            protected_parts = []

            def protect_chinese_quotes(match):
                protected_parts.append(match.group(0))
                return f"__PROTECTED_{len(protected_parts) - 1}__"

            temp_text = re.sub(r'「[^」]*」', protect_chinese_quotes, response_text)
            temp_text = temp_text.replace("'", '"')

            for i, part in enumerate(protected_parts):
                temp_text = temp_text.replace(f"__PROTECTED_{i}__", part)

            response_dict = json.loads(temp_text)
            detail = response_dict.get('detail', [])
            reason = response_dict.get('reason', '')
            if debug:
                print(f"✓ 方法2成功! detail长度: {len(detail)}")
            return detail, reason
        except Exception as e:
            if debug:
                print(f"✗ 方法2失败: {e}")

        # ================= 方法3: 原来的正则提取（修正） =================
        try:
            if debug:
                print("\n尝试方法3: 正则表达式提取")

            reason_match = re.search(
                r"['\"]reason['\"].*?[:\s]+['\"]?([^'\"]+?)['\"]?\s*,?\s*['\"]detail",
                response_text,
                re.DOTALL
            )
            reason = reason_match.group(1).strip() if reason_match else ""

            detail_match = re.search(r"['\"]detail['\"].*?[:\s]+($$[^$$]+\])", response_text)
            if detail_match:
                detail_str = detail_match.group(1)
                detail_str = detail_str.replace(' ', '')
                detail = [int(x.strip()) for x in detail_str.strip('[]').split(',') if x.strip()]

                if debug:
                    print(f"✓ 方法3成功! detail长度: {len(detail)}")
                return detail, reason
        except Exception as e:
            if debug:
                print(f"✗ 方法3失败: {e}")

        # ================= 方法4: 原来激进数组提取（修正） =================
        try:
            if debug:
                print("\n尝试方法4: 激进的数组提取")

            array_matches = re.findall(r'$$[\d,\s]+$$', response_text)
            if array_matches:
                detail_str = array_matches[-1]
                detail = [int(x.strip()) for x in detail_str.strip('[]').split(',') if x.strip()]

                reason = ""
                reason_patterns = [
                    r"reason['\"]?\s*[:：]\s*['\"]?([^'\"]+)",
                    r"评分过程[：:]\s*([^{}$$\]]+)",
                ]
                for pattern in reason_patterns:
                    reason_match2 = re.search(pattern, response_text, re.DOTALL)
                    if reason_match2:
                        reason = reason_match2.group(1).strip()
                        break

                if debug:
                    print(f"✓ 方法4成功! detail长度: {len(detail)}")
                return detail, reason
        except Exception as e:
            if debug:
                print(f"✗ 方法4失败: {e}")

        # ================= 方法5: 宽松模式（支持 reason 裸文本 + 带中文说明的 key） =================
        try:
            if debug:
                print("\n尝试方法5: 宽松解析（reason 裸文本 + 扩展 key）")

            text = response_text

            # 5.1 detail 数组：允许 key 形如 'detail' 或 'detail'（array of Integer）
            detail_match = re.search(
                r"['\"]detail['\"](?:[（(][^）)]*[）)])?\s*[:：]\s*(\[[^$$]*\])",
                text
            )
            if not detail_match:
                # 找不到就退一步：拿最后一个数字数组
                array_matches = re.findall(r'$$[0-9,\s]+$$', text)
                if array_matches:
                    detail_str = array_matches[-1]
                else:
                    if debug:
                        print("✗ 方法5: 没找到 detail 数组")
                    return [], ""
            else:
                detail_str = detail_match.group(1)

            try:
                nums = [int(x.strip()) for x in detail_str.strip('[]').split(',') if x.strip() != ""]
            except Exception as e:
                if debug:
                    print(f"✗ 方法5: 解析 detail 数组失败: {e}")
                nums = []

            # 5.2 reason：允许 'reason' 或 'reason'（String）
            reason = ""
            reason_block_match = re.search(
                r"['\"]reason['\"](?:[（(][^）)]*[）)])?\s*[:：]\s*(.*?)(?=['\"]detail['\"]|$)",
                text,
                flags=re.S
            )
            if reason_block_match:
                reason_raw = reason_block_match.group(1).strip().rstrip(',')
                # 去掉首尾引号（如果有）
                if (reason_raw.startswith('"') and reason_raw.endswith('"')) or \
                        (reason_raw.startswith("'") and reason_raw.endswith("'")):
                    reason_raw = reason_raw[1:-1]
                reason = reason_raw.strip()

            if debug:
                print(f"✓ 方法5成功! detail长度={len(nums)}, reason长度={len(reason)}")

            return nums, reason

        except Exception as e:
            if debug:
                print(f"✗ 方法5失败: {e}")

        # 所有方法都失败
        return [], ""

    except Exception as e:
        if debug:
            print(f"\n解析异常: {e}")
        return [], ""
