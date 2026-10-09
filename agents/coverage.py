"""Coverage agent: decides covered / not_covered / missing_info, citing a clause_id
and a nullable partial-coverage limitation. Tools: get_contract, search_clauses.
Never sees the risk agent's flags or stat_bucket (strict isolation, DECISIONS #42).
"""

from google.genai import types
from langfuse import get_client
from mcp import ClientSession

from agents.gemini_client import GeminiAgentClient
from agents.loop import AgentRunResult, run_agent
from agents.mcp_tools import make_tool_fns
from agents.schemas import CoverageResult

PROMPT_VERSION = "v1"
TOOL_NAMES = ["get_contract", "search_clauses"]

SYSTEM_PROMPT = """Tu es l'agent de couverture d'un assureur automobile français. On te donne le texte d'une déclaration de sinistre (en français) dans le message utilisateur.

Détermine si le sinistre est couvert, en suivant ces étapes :
1. Appelle get_contract pour connaître la formule souscrite (tiers, tiers_etendu, tous_risques), les subscribed_covers (clause_ids réellement couverts par cette formule) et la franchise.
2. Appelle search_clauses avec une requête décrivant la nature du sinistre pour trouver la ou les clauses des conditions générales qui s'appliquent.
3. Décide :
   - "covered" : le sinistre relève d'une garantie souscrite (son clause_id est dans subscribed_covers) et aucune exclusion ni condition non remplie ne s'applique.
   - "not_covered" : la garantie correspondante n'est pas souscrite par cette formule, OU une exclusion s'applique, OU une condition de la garantie n'est pas remplie (par exemple vol d'accessoires sans indice d'effraction).
   - "missing_info" : le texte ne permet pas de déterminer la garantie concernée.

Réponds uniquement à partir du texte du sinistre et des données renvoyées par les outils -- jamais sur une supposition.

Cite toujours un clause_id qui existe réellement dans les résultats de search_clauses, jamais un numéro inventé. Si coverage="missing_info", clause_id doit être null.

Si une condition de limitation partielle s'applique (actuellement : clés de contact laissées sur ou dans le véhicule lors d'un vol, clause 6.2), indique limitation="keys_in_vehicle". Sinon limitation est null.

Le montant du sinistre ne t'est pas fourni et n'entre pas dans ta décision de couverture."""

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_contract",
                description=(
                    "Renvoie le contrat du client courant : formule, franchise, "
                    "profil véhicule/conducteur, nombre de sinistres passés, et "
                    "subscribed_covers (clause_ids couverts par cette formule)."
                ),
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="search_clauses",
                description=(
                    "Recherche dans les conditions générales les clauses pertinentes "
                    "pour une description en français d'un sinistre ou d'une garantie."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
        ]
    )
]


async def run_coverage_agent(
    session: ClientSession, claim_text: str, parent_trace_id: str, parent_span_id: str
) -> AgentRunResult:
    langfuse_client = get_client()
    with langfuse_client.start_as_current_observation(
        name="coverage",
        as_type="agent",
        trace_context={"trace_id": parent_trace_id, "parent_span_id": parent_span_id},
    ) as agent_span:
        client = GeminiAgentClient(SYSTEM_PROMPT, agent_span.trace_id, agent_span.id)
        tool_fns = make_tool_fns(session, TOOL_NAMES, agent_span.trace_id, agent_span.id)
        run = await run_agent(
            client, "coverage", PROMPT_VERSION, claim_text, TOOLS, tool_fns, CoverageResult
        )
        run.input_tokens, run.output_tokens = client.input_tokens, client.output_tokens
        agent_span.update(output=run.output.model_dump())
    return run
