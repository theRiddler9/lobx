"""The service layer: a FastAPI app that starts simulated runs and streams the book.

It only wires components that already exist (the harness, the strategies, the evaluation
statistics); it contains no trading logic of its own.
"""
