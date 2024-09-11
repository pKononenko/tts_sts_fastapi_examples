from openai import OpenAI


class LLMResponder:

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
      Initializes the LLM with the required API key (for OpenAI or other LLM services).
      """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_llm_response(self, user_query: str) -> str:
        """
        Gets a response from the LLM based on the given query.

        Parameters:
        - user_query (str): User prompt string.

        Returns:
        - response (str): LLM answer on user prompt.
        """
        response = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": user_query,
            }],
            model=self.model,
        )
        return response.choices[0].message.content
