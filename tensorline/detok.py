"""Incremental detokenization for streaming.

Re-decoding the whole output on every new token is O(n) per token (O(n^2) per request). Instead,
keep two offsets into the token list (the same approach as vLLM and TGI):
  prefix: start of a small window that has already been emitted (gives the decoder context, so
          e.g. SentencePiece leading spaces and multi-token merges render correctly)
  read:   end of what has been emitted
Each push decodes only ids[prefix:], subtracts the already-emitted ids[prefix:read], and emits
the difference. If the tail is an incomplete UTF-8 sequence (U+FFFD), it waits for more tokens.
"""


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def push(self, new_ids: list[int]) -> str:
        self.ids.extend(new_ids)
        prefix_text = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=True)
        new_text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=True)
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""
