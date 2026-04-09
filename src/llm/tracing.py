import os
from dotenv import load_dotenv

load_dotenv()

_langfuse = None


def get_langfuse():
    global _langfuse
    if _langfuse is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        if not public_key:
            return None
        from langfuse import Langfuse
        _langfuse = Langfuse()
    return _langfuse
