"""Promptfoo custom provider: echoes the prompt back, no API call needed."""


def call_api(prompt, options, context):
    return {"output": prompt}
