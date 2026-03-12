![SAIDA Banner](assets/github-banner.png)

# Semantic Data Context

SAIDA allows datasets to include semantic documentation.

Context files are written in Markdown.

Example:

```
# Dataset: Sales

## Metrics

revenue = total invoice value after discounts

## Important Rules

cancelled orders must be excluded

## Trusted Date Field

posted_at
```

---

## Why Context Matters

Schema alone cannot describe business meaning.

Context provides:

- metric definitions
- business rules
- field explanations
- caveats

This improves NLP request normalization, planning quality, and optional reasoning quality.

It also improves optional LLM interpretation, because LLM proposals are validated against both discovered dataset structure and this semantic context before SAIDA accepts them.

