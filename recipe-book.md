# Mech Interp & ML Python Recipe Book

A living, non-overwhelming cheat sheet of the "beyond intro Python" idioms you keep seeing in PyTorch, ARENA, and TransformerLens code. Each recipe shows the **Intro Python** version you already know right next to the **ML / Research Python** version.

---

## 1. Everyday Python Idioms Used in ML Code

### 1.1 List Comprehensions (1-line `for` + `append`)
**Intro Python:**
```python
valid = []
for country, capital in pairs:
    if len(model.to_str_tokens(" " + capital, prepend_bos=False)) == 1:
        valid.append((country, capital))
```
**ML Python:**
```python
valid = [
    (country, capital)
    for country, capital in pairs
    if len(model.to_str_tokens(" " + capital, prepend_bos=False)) == 1
]
```
*Rule of thumb:* Both do the exact same thing. Read it as: `[ITEM  for  ITEM  in  OLD_LIST  if  CONDITION]`. Use a normal `for` loop whenever a comprehension starts looking messy.

### 1.1b Inspecting Any Function Quickly (No Memorizing!)
- **In VS Code:** Hover your mouse over any function/method, or press `Ctrl+Space` inside autocomplete to expand its docstring.
- **In a Jupyter cell:** Put `?` after any function to print its help (`last_logits.argmax?`), or `??` to see its actual Python source code.
- **3-number dummy test:** Run an unknown method on `x = torch.tensor([1.0, 5.0, 2.0])` and print `out, type(out), out.shape`.

### 1.2 `enumerate()` and `zip()` (Avoiding `range(len(...))`)
- **`enumerate(items)`** gives you `(index, item)` together.
- **`zip(list_a, list_b)`** walks through two lists side by side.
```python
for i, (country, capital) in enumerate(pairs):
    print(i, country, capital)

for prompt, answer in zip(prompts, answers):
    print(prompt, "->", answer)
```

### 1.3 Tuple Unpacking & `_` (Ignore Variable)
Functions in ML often return 2+ things at once. `_` is just a normal variable name people use to signal "I don't care about this value."
```python
logits, cache = model.run_with_cache(prompt)   # unpacks 2 return values
_, cache = model.run_with_cache(prompt)        # ignores logits, keeps cache
```

### 1.4 f-Strings with Number Formatting
Instead of `print("Prob:", round(prob * 100, 1), "%")`:
```python
print(f"{country:12s} -> {capital:10s} | prob={prob:.1%} | logit={logit:.2f}")
# :12s  = pad text to 12 spaces (makes clean aligned tables!)
# :.1%  = multiply by 100 and show 1 decimal with % (0.284 -> 28.4%)
# :.2f  = show 2 decimal places (14.8123 -> 14.81)
```

### 1.5 Type Hints (Just Readable Labels)
You will often see `: torch.Tensor` or `-> float` in function headers:
```python
def get_logit(prompt: str, target: str) -> float:
    ...
```
Python ignores these at runtime. They are just notes telling humans (and autocomplete) what type goes in and comes out.

---

## 2. PyTorch Tensors (Why ML Code Skips `for` Loops)

A PyTorch **tensor** is basically a grid of numbers (1D = list, 2D = table, 3D = stack of tables) that lives on the GPU and does math on **every number at once** (much faster than a Python loop).

### 2.1 Always Check `.shape` First!
When confused in PyTorch, print `.shape`.
```python
logits = model(prompt)
print(logits.shape)   # torch.Size([1, 6, 50257]) -> (batch, seq_len, vocab_size)
```

### 2.2 Slicing with `:` and `-1`
- `:` means "keep everything along this dimension."
- `-1` means "the last element."
```python
last_logits = logits[0, -1, :]   # prompt 0, last token, all 50257 vocab scores
# (trailing ":" can be omitted, so logits[0, -1] means the exact same thing!)
```

### 2.3 Boolean Masks Instead of Counting Loops
**Intro Python:**
```python
count = 0
for score in last_logits:
    if score > last_logits[target_id]:
        count += 1
```
**PyTorch:**
```python
rank = (last_logits > last_logits[target_id]).sum().item()
```
*How it works:* `last_logits > x` produces a tensor of `True`/`False` values (`True=1`, `False=0`). `.sum()` adds up all the `True`s, and `.item()` converts the 1-number tensor to a normal Python `int`.

### 2.4 `dim=-1` vs. `dim=0` (Concrete 2x3 Example)
Whenever a tensor has multiple dimensions, `dim` tells PyTorch **which axis collapses or normalizes** (`softmax`, `sum`, `mean`, `argmax`).

Suppose you have **2 prompts** (rows, `dim=0`) and **3 words** (columns, `dim=1` / `dim=-1`):
```python
scores = torch.tensor([
    [10.0,  2.0,  2.0],   # Prompt 0
    [ 1.0,  9.0,  1.0],   # Prompt 1
])
torch.softmax(scores, dim=-1)  # Each ROW sums to 1 (probabilities per prompt)
torch.softmax(scores, dim=0)   # Each COLUMN sums to 1 (compares prompts - usually a bug!)
scores.mean(dim=0)             # Averages DOWN columns (mean score across all prompts)
```

### 2.4b `max`, `argmax`, and `topk`
- **`x.max()`** returns the biggest **value**.
- **`x.argmax()`** returns the **index** (position) of the biggest value.
- **`x.topk(k=5)`** returns **two** tensors unpacked as a tuple: `(top_values, top_indices)`.
- Single-tensor ops work as **both** a method (`x.argmax()`) and a function (`torch.argmax(x)`). Creating/combining tensors uses `torch.` functions (`torch.tensor(...)`, `torch.stack(...)`).
```python
top_probs, top_ids = probs.topk(k=5)   # ALWAYS unpack both!
if target_id in top_ids:
    print("Target is in top 5!")
```

### 2.5 Moving Between Python & PyTorch
```python
x.item()                  # 1-number tensor -> normal Python int/float
torch.stack(tensor_list)  # list of same-shape tensors -> one combined tensor
utils.to_numpy(tensor)    # GPU tensor -> NumPy array (for plotting)
torch.set_grad_enabled(False)  # turns off training gradients (saves RAM & speed)
```

### 2.6 `einops.einsum` (Readable Matrix Math)
Instead of cryptic `.T`, `@`, and `torch.matmul` where you forget which axis multiplies which, mech interp code names every dimension explicitly:
```python
import einops
# Multiply each layer's residual vector [layer, d_model] with one unembed vector [d_model]
# Result has shape [layer] (d_model gets multiplied and summed away!)
logits_by_layer = einops.einsum(resid_stack, unembed_dir, "layer d_model, d_model -> layer")
```
*How to read it:* Dimensions on the left that disappear on the right get multiplied and summed (dot product).

---

## 3. TransformerLens 4.0 Essentials (Added as You Go)

### Model Loading & Tokenization
```python
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens import utilities as utils

model = TransformerBridge.boot_transformers("gpt2", device=device)
model.enable_compatibility_mode()

# Leading space matters! (" Paris", not "Paris")
tokens = model.to_tokens(" Canberra", prepend_bos=False)   # text -> tensor of token IDs
target_id = model.to_single_token(" Paris")                # 1-token text -> int ID (errors if 2+ tokens)
pieces = model.to_str_tokens(" Canberra", prepend_bos=False)  # text or IDs -> list of token strings
word = model.to_string(target_id)                          # 1 ID or IDs -> one joined string
utils.test_prompt("The capital of France is", " Paris", model)
```

### Core Mech Interp Concepts So Far
- **Logit Lens:** Check the residual stream's prediction *after each layer* (when does the answer emerge?).
- **Direct Logit Attribution (DLA):** Split the final logit into each Attention and MLP layer's direct dot product with `W_U` (who wrote the answer?).
- **DLA Caveats:** Misses indirect effects (early heads feeding later MLPs), freezes final LayerNorm scale, and does not prove causal necessity (backup circuits).
- **Red-Teaming:** Always test prompt rephrasings and subtract a baseline (`logit(correct) - mean(logit(other capitals))`) so generic category boosts cancel out.
