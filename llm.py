"""
llm.py — Conversation with Claude Haiku via the Anthropic API.

Maintains a running conversation history for multi-turn dialogue.
Call set_user_name(name) after onboarding to personalise the system prompt.
Call clear_history() to start a fresh conversation.
"""

import anthropic

from config import ANTHROPIC_API_KEY
from rag import get_relevant_context

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Conversation history: list of {"role": "user"|"assistant", "content": str}
_history:   list[dict] = []
_user_name: str        = ""

_BASE_SYSTEM = """
أنت آيمي، المساعد الشخصي الذي يمثّل عبدالعزيز الحيدان ويتحدث بالنيابة عنه.
مهمتك أن ترد على أي شخص يسأل عن عبدالعزيز — سواء عن مهاراته أو تجاربه أو مشاريعه أو شخصيته.

قواعد أساسية:
- تحدث بضمير المتكلم عن عبدالعزيز من منظور ممثله، مثل: "عبدالعزيز يتقن..." أو "من أبرز نقاط قوته..."
- ركّز دائماً على نقاط القوة والإنجازات والصفات الإيجابية.
- إذا سُئلت عن نقطة ضعف، حوّلها إلى فرصة تطوير بأسلوب إيجابي.
- ردودك قصيرة ومناسبة للمحادثة الصوتية — جملتان إلى أربع كحد أقصى.
- تحدث بالعربية افتراضياً، وبالإنجليزية إذا خاطبك الشخص بالإنجليزية.
- كن دافئاً وواثقاً وطبيعياً. تجنب القوائم والتنسيق.
- لا تذكر أنك ذكاء اصطناعي ما لم يُسأل صراحةً.
"""


def _build_system(context: str = "") -> str:
    """Return the system prompt with RAG context injected if available."""
    system = _BASE_SYSTEM
    if context:
        system += (
            "\n\nمعلومات ذات صلة عن عبدالعزيز يمكنك الاستعانة بها إذا كانت مرتبطة بالسؤال:\n"
            + context
        )
    return system


def set_user_name(name: str) -> None:
    """Call this after onboarding to personalise Claude's responses."""
    global _user_name
    _user_name = name


def get_response(user_message: str) -> str:
    """
    Send user_message to Claude Haiku and return the assistant's reply.
    Conversation history is preserved across calls.
    RAG context is injected into the system prompt per turn.
    """
    context = get_relevant_context(user_message)
    _history.append({"role": "user", "content": user_message})

    response = _client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=_build_system(context),
        messages=_history,
    )

    reply = response.content[0].text.strip()
    _history.append({"role": "assistant", "content": reply})
    return reply


def clear_history() -> None:
    """Reset the conversation history (keeps the user name)."""
    _history.clear()

