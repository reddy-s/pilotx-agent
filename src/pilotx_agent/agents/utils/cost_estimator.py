class TokenUsage:
    @staticmethod
    def compute_token_cost(
        prompt_tokens: int,
        completion_tokens: int,
        cost_per_1m_input_tok: float = 0.15,
        cost_per_1m_output_tok: float = 0.60,
    ) -> tuple[float, float, float]:
        """
        Compute the cost based on the number of prompt and completion tokens.

        Args:
            prompt_tokens (int): Number of tokens in the prompt.
            completion_tokens (int): Number of tokens in the completion.
            cost_per_1m_input_tok (float): Cost per 1 million input tokens. Default is $0.1.
            cost_per_1m_output_tok (float): Cost per 1 million output tokens. Default is $0.2.

        Returns:
            float: Estimated cost.
        """
        input_cost = (prompt_tokens / 1e6) * cost_per_1m_input_tok
        output_cost = (completion_tokens / 1e6) * cost_per_1m_output_tok
        total_cost = input_cost + output_cost
        return total_cost, input_cost, output_cost

    @staticmethod
    def get_used_context_length(
        total_tokens, max_context_length: int = 128000, buffer: int = 0.1
    ) -> float:
        return round(total_tokens / (max_context_length * (1 - buffer)), 4)
