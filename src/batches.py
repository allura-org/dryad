import llama_cpp
import ctypes

class Batch:
    def __init__(self, n_tokens: int, n_ctx: int = 0):
        self.n_tokens = n_tokens
        self.n_ctx = n_ctx

        self.batch = llama_cpp.llama_batch_init(n_tokens, 0, n_ctx)
    
    def __del__(self):
        llama_cpp.llama_batch_free(self.batch)
    
    def common_batch_add(
            self,
            id: llama_cpp.llama_token,
            pos: llama_cpp.llama_pos,
            seq_ids: ctypes.Array[llama_cpp.llama_seq_id],
            logits: bool
    ) -> None:
        self.batch.token[self.batch.n_tokens] = id
        self.batch.pos[self.batch.n_tokens] = pos
        self.batch.n_seq_id[self.batch.n_tokens] = len(seq_ids)
        for i in range(len(seq_ids)):
            self.batch.seq_id[self.batch.n_tokens][i] = seq_ids[i]
        self.batch.logits[self.batch.n_tokens] = int(logits)

        self.batch.n_tokens += 1

    def common_batch_clear(self) -> None:
        self.batch.n_tokens = 0

    def set_tokens(self, tokens: list[llama_cpp.llama_token], last_logits: bool = True) -> None:
        for i in range(len(tokens)):
            self.common_batch_add(tokens[i], self.batch.n_tokens, [1, 0], False)
        if last_logits:
            self.batch.logits[self.batch.n_tokens - 1] = True

    def eval(self, ctx: llama_cpp.llama_context_p) -> None:
        res = llama_cpp.llama_decode(ctx, self.batch)
        if res != 0:
            raise RuntimeError(f"llama_decode returned {res}")
