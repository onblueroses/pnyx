# Prompt Templates

Fill these in once you've locked the project idea (8-hour mark).

## Template: System Prompt

```
You are [role] helping [user type] [do what].

Given [input type], you:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Output format:
[describe the structured output]

Rules:
- [constraint 1]
- [constraint 2]
```

## Template: Chain Step

For multi-step pipelines, each step should have its own focused prompt.

```
Step: [name]
Input: [what comes in]
Task: [single focused task]
Output: [exact format]
```

## Notes

- System prompts are the highest-leverage thing you control. Spend time here.
- Use structured output (JSON mode or XML tags) so downstream steps parse reliably.
- Test each step in isolation before chaining.
