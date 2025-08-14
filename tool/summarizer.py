from transformers import pipeline

def summarize_chat(chat_history: str) -> str:
    """
    Summarizes a chat history using a distilled BART model.

    Args:
        chat_history (str): The full chat transcript as a string.

    Returns:
        str: A neatly formatted summary.
    """
    try:
        # Use a smaller, more efficient model for summarization
        summarizer = pipeline("summarization", model="ibm-granite/granite-3.3-2b-instruct", device_map="auto")

        # Generate the summary
        # Ensure the input text is not excessively long for the model
        max_input_length = summarizer.tokenizer.model_max_length
        truncated_history = chat_history[:max_input_length]

        summary_result = summarizer(
            truncated_history,
            max_length=150,       # Max length of the summary
            min_length=30,        # Min length of the summary
            do_sample=False
        )

        return summary_result[0]['summary_text']

    except Exception as e:
        return f"Error during summarization: {e}"
