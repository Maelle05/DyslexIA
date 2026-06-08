"""Generate short French reading passages with Claude Haiku.

Each passage is fresh and randomized so participants never re-read a memorized
text. Length is targeted so a natural read comfortably exceeds the 999-sample
(~20 s) recording floor even for fast readers, while a slow read simply caps at
the 1999-sample (~40 s) ceiling. Falls back to a built-in passage bank when no
API key is configured, so the app remains usable offline.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

HAIKU_MODEL = "claude-haiku-4-5"

# Topics steer variety; the model still writes freely within each.
_TOPIC_SEEDS = (
    "un renard curieux dans une forêt en hiver",
    "le premier matin dans une ville au bord de la mer",
    "un enfant qui apprend à faire du vélo",
    "un vieil horloger et son atelier",
    "un jardin qui s'éveille au printemps",
    "un gardien de phare pendant une tempête",
    "deux amis qui construisent une cabane dans un arbre",
    "un boulanger qui ouvre sa boutique avant l'aube",
    "un voyage dans un train de montagne très lent",
    "un chat qui explore une bibliothèque tranquille",
)

_FALLBACK_PASSAGES = (
    "La petite barque glissait sur la rivière calme tandis que le soleil se "
    "levait lentement derrière les collines. Des oiseaux chantaient parmi les "
    "hauts roseaux, et l'eau brillait comme du verre. Un pêcheur salua depuis la "
    "berge, sa ligne déjà jetée dans le doux courant. C'était le genre de matin "
    "qui donnait à toute la vallée un air de renouveau. Plus loin, un héron se "
    "tenait parfaitement immobile au milieu des joncs, attendant avec une "
    "patience infinie qu'un poisson passe sous la surface. La barque s'éloigna "
    "sans le moindre bruit, et le vieil homme aux rames sourit pour lui-même, "
    "heureux de laisser la rivière le porter où elle voulait. Il n'avait aucun "
    "horaire à tenir et aucune raison de se presser, seulement la large eau "
    "tranquille et la lumière douce qui s'étendait sur les champs.",
    "Chaque soir, le vieux boulanger balayait le sol et laissait reposer la "
    "pâte. L'odeur chaude du pain emplissait la rue étroite au-dehors. Les "
    "enfants collaient leur visage à la vitrine, espérant un petit pain tout "
    "frais. Il en gardait toujours quelques-uns, juste pour les voir sourire en "
    "rentrant chez eux au crépuscule. Au matin, les fours brillaient de nouveau, "
    "et des plateaux de miches dorées refroidissaient sur les étagères en bois "
    "près du comptoir. Les voisins arrivaient un à un, attirés par le parfum qui "
    "les accueillait chaque jour depuis trente ans. Le boulanger connaissait "
    "chacun par son nom, et il prenait des nouvelles de leur famille en "
    "enveloppant leur pain dans du papier propre. C'était un travail simple, "
    "mais il donnait à toute la rue son rythme paisible et son premier petit "
    "réconfort de la journée.",
)


@dataclass
class PassageRequest:
    """Parameters controlling passage generation."""

    word_count: int = 130
    reading_level: str = "enfant (9 - 10 ans) tout public"
    topic: str | None = None


def _build_prompt(request: PassageRequest) -> str:
    topic = request.topic or random.choice(_TOPIC_SEEDS)
    return (
        f"Rédige un seul texte de lecture autonome d'au moins "
        f"{request.word_count} mots (assez long pour que sa lecture à voix haute "
        f"dépasse 25 secondes) sur le sujet suivant : {topic}. Niveau de lecture "
        f"du public : {request.reading_level}. Utilise une prose claire et "
        f"naturelle en phrases complètes, sans titre, sans liste, sans "
        f"guillemets. N'écris que le texte du passage sans titre, en français."
    )


def generate_passage(
    request: PassageRequest | None = None,
    api_key: str | None = None,
) -> str:
    """Return a fresh reading passage, or a fallback if the API is unavailable."""
    request = request or PassageRequest()
    resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        return random.choice(_FALLBACK_PASSAGES)

    try:
        import anthropic
    except ImportError:
        return random.choice(_FALLBACK_PASSAGES)

    client = anthropic.Anthropic(api_key=resolved_api_key)
    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=600,
        temperature=1.0,
        messages=[{"role": "user", "content": _build_prompt(request)}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
