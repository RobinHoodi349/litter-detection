from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from litter_detection.config import Settings
from litter_detection.agent.models import VerifiedDetection, VerifierDeps

settings = Settings()

provider = OpenAIProvider(base_url="https://localhost:11434/v1", api_key="ollama")
vision_model = OpenAIChatModel(settings.VISION_MODEL_NAME, provider=provider)

verifier_agent = Agent(
    vision_model,
    deps_type=VerifierDeps,
    result_type=VerifiedDetection,
    system_prompt=(
        "You are a litter detection quality controller for an autonomous robot dog "
        "patrolling an outdoor area. "
        "The robot's ML segmentation model has flagged a camera frame as potentially "
        "containing litter. Your job is to visually inspect the image and confirm or "
        "reject the detection. "
        "The image you receive has red-highlighted pixels overlaid on it — these mark "
        "exactly the areas the ML model classified as litter. Focus your assessment on "
        "these highlighted regions. "
        "Litter includes: plastic bottles, bags, wrappers, cans, paper, cigarette butts, "
        "or any other human-made waste left in the environment. "
        "Do NOT flag: leaves, sticks, mud, puddles, shadows, or natural debris. "
        "Be precise — false positives cause unnecessary robot stops."
    ),
)
