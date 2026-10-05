# Decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | `google-genai` (not `google-generativeai`) | New unified SDK; supports Gemini 2.0+ and function calling in one package |
| 2 | `mcp[cli]` extra | Adds the `mcp` dev CLI for inspecting servers during development |
| 3 | `dependency-groups` for dev deps | uv native; keeps test/lint tools out of the production env |
| 4 | `asyncio_mode = "auto"` in pytest | MCP and A2A are async throughout; avoids boilerplate `@pytest.mark.asyncio` on every test |
| 5 | `promptfoo@0.115.0` pinned (not latest) | Node 22.18.0 is installed; 0.122.0+ requires 22.22.0 and hard-exits; 0.120.0 has a Drizzle migration bug; 0.115.0 is the newest clean version |
