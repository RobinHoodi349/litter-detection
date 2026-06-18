from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from litter_detection.config import Settings
from litter_detection.agent.models import VerifiedDetection, VerifierDeps

settings = Settings()

provider = OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama")
vision_model = OpenAIChatModel(settings.VISION_MODEL_NAME, provider=provider)

verifier_agent = Agent(
    vision_model,
    deps_type=VerifierDeps,
    output_type=VerifiedDetection,
    system_prompt=(
        "You are a litter detection quality controller for an autonomous robot dog "
        "patrolling an outdoor area.\n\n"
        "You will receive a CROPPED image zoomed into the region the robot's ML "
        "segmentation model flagged. Red-highlighted pixels mark exactly what the model detected. "
        "Your job: decide whether those red pixels show real human-made litter.\n\n"
        "CONFIRM as litter (litter_confirmed=true):\n"
        "- Plastic bottles, cups, bags, packaging, film\n"
        "- Food wrappers, fast-food containers, aluminium cans, glass bottles\n"
        "- Paper, cardboard, tissues, cigarette butts\n"
        "- Any other man-made waste clearly out of place in the environment\n\n"
        "REJECT as not litter (litter_confirmed=false):\n"
        "- Leaves, grass, sticks, branches, bark, pine cones\n"
        "- Mud, puddles, wet ground, shadows, dirt patches\n"
        "- Rocks, pebbles, gravel, natural ground texture\n"
        "- Parts of the robot, camera lens flare, image artefacts\n\n"
        "Be strict: only confirm when you are reasonably sure the highlighted area "
        "shows actual litter. When in doubt, reject. "
        "False positives cause unnecessary robot stops; false negatives are less costly."
    ),
)
