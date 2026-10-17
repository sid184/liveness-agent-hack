"""The agent: issue a challenge -> observe -> perceive (YOLO) -> reason (VLM) -> score -> record.

agent/llm.py   W&B-hosted text model used for challenge choice and verdict writing.
agent/loop.py  one session end to end. First version is deliberately naive; Karim owns scoring.
"""
