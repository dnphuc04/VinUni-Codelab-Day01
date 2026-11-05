"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    
    #Solution:
    from openai import OpenAI

    # 1. Khởi tạo OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 2. Bắt đầu đo thời gian
    start_time = time.perf_counter()

    # 3. Gửi request đến OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # 4. Tính thời gian phản hồi
    latency = time.perf_counter() - start_time

    # 5. Lấy nội dung phản hồi
    response_text = response.choices[0].message.content or ""

    # 6. Thống kê token
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    # 7. Trả kết quả
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
    
    #Solution:
    from google import genai
    from google.genai import types

    # 1. Khởi tạo Gemini Client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 2. Cấu hình các tham số
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    # 3. Bắt đầu đo thời gian
    start_time = time.perf_counter()

    # 4. Gửi request đến Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # 5. Tính thời gian phản hồi
    latency = time.perf_counter() - start_time

    # 6. Lấy nội dung phản hồi
    response_text = response.text or ""

    # 7. Lấy thông tin token usage
    metadata = response.usage_metadata

    usage = {
        "input_tokens": metadata.prompt_token_count or 0,
        "output_tokens": metadata.candidates_token_count or 0,
    }

    # 8. Trả về kết quả
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
    # raise NotImplementedError("Implement call_anthropic")

    #Solution:
    import anthropic

    # 1. Khởi tạo Anthropic Client bằng API key từ biến môi trường
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # 2. Bắt đầu đo thời gian gọi API
    start_time = time.perf_counter()

    # 3. Gửi câu hỏi và các tham số cấu hình đến Claude
    response = client.messages.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # 4. Tính thời gian phản hồi, đơn vị là giây
    latency = time.perf_counter() - start_time

    # 5. Ghép nội dung các khối văn bản thành một chuỗi
    # Claude có thể trả về nhiều khối nội dung trong response.content
    response_text = "".join(
        block.text
        for block in response.content
        if isinstance(getattr(block, "text", None), str) # Chỉ lấy thuộc tính text khi giá trị thực sự là chuỗi
    )

    # 6. Lấy số token đầu vào và đầu ra
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    # 7. Trả kết quả theo cùng cấu trúc với OpenAI và Gemini
    return response_text, latency, usage

# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.
    
    #Solution:
    # 1. Khai báo các model cần so sánh
    # Mỗi phần tử gồm: tên kết quả, hàm gọi API, tên model
    models = [
        ("gpt4o", call_openai, OPENAI_MODEL),
        ("gpt4o_mini", call_openai, OPENAI_MINI_MODEL),
        ("gemini_flash", call_gemini, GEMINI_MODEL),
    ]

    # 2. Tạo dictionary để lưu kết quả của tất cả model
    results = {}

     # 3. Lần lượt gửi cùng một câu hỏi đến từng model
    for name, call_fn, model in models:
        response_text, latency, usage = call_fn(
            prompt,
            model=model,
        )

        # 4. Lấy số token đã sử dụng
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        # 5. Tra đơn giá trong bảng giá có sẵn của bài tập
        # Đơn giá được tính bằng USD trên mỗi 1 triệu token
        pricing = PRICING_1M_TOKENS[model]

        # 6. Tính tổng chi phí cho token đầu vào và đầu ra
        # Chi phí = (token vào × giá vào + token ra × giá ra) / 1 triệu
        cost = (
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

        # 7. Lưu nội dung, thời gian, chi phí và số token của model
        results[name] = {
            "response": response_text,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    # 8. Trả về bảng kết quả để có thể so sánh các model
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.
    #raise NotImplementedError("Implement streaming_chatbot")

    #Solution:
    from google import genai
    from google.genai import types

    # 1. Khởi tạo Gemini Client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # 2. Tạo danh sách lưu lịch sử hội thoại
    # Một lượt hội thoại gồm một tin nhắn user và một tin nhắn model
    history = []

    # 3. Lặp liên tục để người dùng có thể hỏi nhiều câu
    while True:
        # 4. Đọc câu hỏi và xử lý trường hợp người dùng ngắt chương trình
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKết thúc.")
            break

        # 5. Dừng chatbot khi người dùng nhập quit hoặc exit
        if user_input.lower() in {"quit", "exit"}:
            print("Kết thúc.")
            break

        # 6. Bỏ qua câu hỏi rỗng và quay lại bước nhập
        if not user_input:
            continue

        # 7. Chuyển câu hỏi thành định dạng nội dung của Gemini
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        # 8. Tạo danh sách để ghép các đoạn trả lời nhận được từ stream
        answer_parts = []
        print("Gemini: ", end="", flush=True)

        try:
            # 9. Gửi lịch sử hội thoại cùng câu hỏi hiện tại đến Gemini
            # API streaming trả về từng đoạn thay vì chờ toàn bộ câu trả lời
            stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=history + [user_message],
            )

            # 10. Hiển thị từng đoạn ngay khi nhận được
            # Đồng thời lưu lại để tạo câu trả lời đầy đủ
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    answer_parts.append(chunk.text)

            # Xuống dòng sau khi Gemini trả lời xong
            print()

        except Exception as exc:
            # Nếu gọi API thất bại, thông báo lỗi và cho phép hỏi lại
            # Lượt bị lỗi chưa được thêm vào lịch sử
            print(f"\nLỗi gọi Gemini: {exc}")
            continue

        # 11. Ghép các đoạn thành câu trả lời đầy đủ
        response_text = "".join(answer_parts)

        # 12. Chuyển câu trả lời thành tin nhắn có role là model
        model_message = types.Content(
            role="model",
            parts=[types.Part(text=response_text)],
        )

        # 13. Thêm cả câu hỏi và câu trả lời vào lịch sử
        history.extend([user_message, model_message])

        # 14. Chỉ giữ lại 3 lượt hội thoại gần nhất
        # 3 lượt × 2 tin nhắn = 6 tin nhắn
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """

    #Solution:
    # 1. Kiểm tra cấu hình trước khi thực hiện lời gọi
    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries và base_delay phải không âm")

    # 2. Gọi lần đầu, sau đó thử lại tối đa max_retries lần
    for attempt in range(max_retries + 1):
        try:
            # 3. Trả kết quả ngay khi hàm chạy thành công
            return fn()
        except Exception:
            # 4. Nếu đã hết lượt thử lại, phát sinh lại lỗi cuối cùng
            if attempt == max_retries:
                raise

            # 5. Chờ trước khi thử lại: base_delay, gấp đôi, gấp bốn, ...
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """

    #Solution:
    # 1. Tạo danh sách chứa kết quả của từng câu hỏi
    results = []

    # 2. So sánh các model với từng câu hỏi trong danh sách
    for prompt in prompts:
        comparison = compare_models(prompt)

        # 3. Lưu câu hỏi gốc cùng kết quả, không sửa dictionary ban đầu
        results.append({**comparison, "prompt": prompt})

    # 4. Trả về toàn bộ kết quả theo đúng thứ tự câu hỏi
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """

    #Solution:
    # 1. Ánh xạ khóa kết quả sang tên model dễ đọc
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }

    # 2. Chuẩn hóa nội dung để dấu | và xuống dòng không làm vỡ bảng
    def escape_cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    # 3. Tạo tiêu đề và dòng phân cách của bảng Markdown
    rows = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    # 4. Mỗi câu hỏi tạo một dòng cho mỗi model
    for result in results:
        prompt = escape_cell(result["prompt"])
        for model_key, stats in result.items():
            if model_key == "prompt":
                continue

            # 5. Rút gọn câu trả lời còn tối đa 50 ký tự, kể cả dấu ...
            response = stats["response"]
            if len(response) > 50:
                response = response[:47] + "..."
            response = escape_cell(response)
            model_name = escape_cell(model_names.get(model_key, model_key))

            # 6. Định dạng thời gian, số token và chi phí cho từng dòng
            rows.append(
                f"| {prompt} | {model_name} | {response} | "
                f"{stats['latency']:.3f}s | "
                f"{stats['input_tokens']} / {stats['output_tokens']} | "
                f"{stats['cost']:.8f} |"
            )

    # 7. Ghép các dòng thành chuỗi Markdown hoàn chỉnh
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
