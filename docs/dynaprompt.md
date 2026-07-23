<p align="center"><img src="https://raw.githubusercontent.com/mohamed-em2m/dynaprompt/main/art/dynaprompt.png" alt="dynaprompt logo"></p>

> **dynaprompt** - Dynamic prompt management and configuration library for LLM applications. Powerful, lazy-loading, and supports Jinja2 templates and Pydantic schemas.

[![MIT License](https://img.shields.io/badge/license-MIT-007EC7.svg?style=flat-square)](https://github.com/mohamed-em2m/dynaprompt/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/dynaprompt.svg)](https://pypi.org/pypi/dynaprompt)
[![Code Style Black](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-75%25-green.svg)](https://github.com/mohamed-em2m/dynaprompt/tree/main/tests)

DynaPrompt is a powerful, lazy-loading prompt configuration manager inspired by **Dynaconf**. It offers a structured way to manage, version, and render LLM prompts while keeping your templates completely separated from your application logic.

---

<p align="center">
  <a href="#-the-problem">The Problem</a> •
  <a href="#-features">Features</a> •
  <a href="#-usage-examples">Usage Examples</a> •
  <a href="#-validation--hooks">Validation & Hooks</a>
</p>

---

## 🛑 The Problem: Why DynaPrompt?

Writing LLM apps usually starts simple, but quickly becomes an unmaintainable mess of hardcoded strings, `f-strings`, and scattered configuration dictionaries.

### ❌ Without DynaPrompt (The Mess)
```python
import os, json

# Hardcoded, mixed with logic, impossible to swap easily for testing/production
SYSTEM_PROMPT = f"""
You are a helpful assistant.
Current User: {user_name}
Format your output according to this schema:
{json.dumps(MySchema.model_json_schema())}
"""

if os.getenv("ENV") == "production":
    model = "gpt-4"
    temperature = 0.2
else:
    model = "gpt-3.5-turbo"
    temperature = 0.7

response = llm_client.generate(prompt=SYSTEM_PROMPT, model=model, temp=temperature)
```

### ✅ With DynaPrompt (Clean & Maintainable)
```python
from dynaprompt import DynaPrompt

# Zero I/O at import. Auto-discovers environments, schemas, and templates.
prompts = DynaPrompt(settings_files=["prompts/"])

# Automatically uses the right model/temp for your current environment!
rendered = prompts.system.render(user_name="Emam")

response = llm_client.generate(
    prompt=rendered.text,
    model=rendered.config["model"],
    temp=rendered.config["temperature"],
    response_format=rendered.response_schema
)
```

---

## 🚀 Installation

```bash
# Using pip
pip install dynaprompt

# Using uv (recommended)
uv add dynaprompt
```

---

## 📖 Usage Examples

### 1. Markdown with YAML Frontmatter (The Cleanest Way)
DynaPrompt allows you to define prompts as standalone Markdown files. You can attach LLM configuration (like `model`, `temperature`, or required `response_schema`) directly at the top of the file using YAML Frontmatter.

**`prompts/analyzer.md`**
```markdown
---
model: gpt-4o
temperature: 0.2
max_tokens: 1000
response_schema: AnalysisSchema
---
You are an expert code analyzer.
Please review the following code snippet from {{ developer_name }}:

\```python
{{ code_snippet }}
\```

Analyze it and return the result strictly matching the schema.
```

**Usage:**
```python
prompts = DynaPrompt(settings_files=["prompts/"])

# Renders the Jinja template with your variables
rendered = prompts.analyzer.render(developer_name="Emam", code_snippet="print('hello')")

print(rendered.config["model"]) # "gpt-4o"
```

### 2. Environment Layering (Dev vs Prod)
You can define base settings and then override them for specific environments (e.g., `development`, `production`). DynaPrompt automatically switches based on `ENV_FOR_DYNAPROMPT`.

**`prompts.toml`**
```toml
# Base settings for all environments
[default.summarizer]
template = "Summarize this: {{ text }}"
model = "gpt-3.5-turbo"
temperature = 0.7

# Overrides for production ONLY
[production.summarizer]
model = "gpt-4-turbo"
temperature = 0.1
```

**Usage:**
```python
# Default environment
prompts = DynaPrompt(settings_files=["prompts.toml"], env="development")
print(prompts.summarizer.config["model"])  # "gpt-3.5-turbo"

# Switch to production dynamically
with prompts.using_env("production"):
    print(prompts.summarizer.config["model"])  # "gpt-4-turbo"
```

### 3. File-Based Templates and Variables
Keep your configuration files pristine. DynaPrompt can automatically resolve templates and variables from external files or Python modules.

```toml
[default.customer_service]
# Load text directly from an external markdown file
template = "prompts/customer_service.md"

# Dynamically import a string variable from a Python file!
# (Extracts 'greeting_prompt' from 'config/prompts.py')
fallback_template = "config.prompts.greeting_prompt"

# Merge specific dictionaries or load entire Python modules as global variables
variables = [
    "config/settings.json",
    "myapp.config:constants"
]
```

### 4. Auto-Exporting Prompts to TOML
You can automatically export your entire loaded prompt structure into a central `pyprompts.toml` file. To keep things clean and optimized:
- **Multiline Templates**: Saved as separate `.md` files in a `prompts/` directory.
- **TOML**: References these files by relative path.

```python
prompts = DynaPrompt(settings_files=["examples/"], auto_export=True)
_ = prompts.google.gemini # Triggers lazy-load and export
```

---

## 🛡️ Validation & Hooks

DynaPrompt allows you to enforce constraints on your rendered prompts (Validation) and intercept the rendering process to inject context automatically (Hooks).

### Example: Enforcing Token Limits and Injecting Context

```python
from dynaprompt import DynaPrompt
from dynaprompt.validator import PromptValidator

# 1. Create a Validator to prevent overly long prompts
class TokenLimitValidator(PromptValidator):
    def validate(self, node, rendered) -> None:
        if len(rendered.text.split()) > 2000:
            raise ValueError(f"Prompt '{node.name}' exceeds maximum token length!")

prompts = DynaPrompt(
    settings_files=["prompts/"],
    validators=[TokenLimitValidator()]
)

# 2. Add a Pre-Render Hook to automatically inject the current date into EVERY prompt
def inject_date(node, kwargs):
    from datetime import datetime
    kwargs["current_date"] = datetime.now().strftime("%Y-%m-%d")
    return kwargs

prompts.add_hook("before_render", "inject_date", inject_date)

# 3. Render
# The hook automatically injects `current_date`, and the validator ensures it's safe!
rendered = prompts.system.render(user_name="Emam")
```

---

## 🔍 Inspection & Tab-Completion
DynaPrompt is designed for developer productivity.
- **Tab-Completion**: Use `dir(prompts)` or hit `Tab` in your IDE to see all available prompts and schemas.
- **History Tracking**: Inspect exactly where a prompt was loaded from and how it was merged across layers.

```python
print(prompts.inspect("customer_support"))
```
