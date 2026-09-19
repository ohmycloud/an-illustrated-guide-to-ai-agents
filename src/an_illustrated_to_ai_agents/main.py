import json
import urllib.request
from dataclasses import dataclass


@dataclass
class Response:
    """Structured response from LLM calls."""

    content: str = ""
    reasoning: str | None = None
    tool_call: dict | None = None
    metadata: dict | None = None


@dataclass
class Step:
    """A single step in an agent's trajectory."""
    thought: str = ""
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict | None = None


class Trajectory:
    """Records agent execution as a sequence of runs."""
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """Register a new run with the given query."""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None = None) -> None:
        """Record a step from a Response, optionally with an observation."""
        # Add THOUGHT
        step = Step(
            thought=response.reasoning or "",
            metadata=response.metadata,
        )

        # Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)


class LLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
    ):
        """Initialize the LLM with the given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think

    def generate(
        self, messages: list[dict], tools: list | None = None
    ) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages,
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools

        if not self.think:
            body["reasoning_effort"] = "none"

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"]
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata,
        )


class TinyAgent:
    """A minimal, modular, and educational agent framework."""
    def __init__(self, llm: LLM):
        self.llm = llm
        self.memory  = None  # Add Memory
        self.tools   = None  # Add Tools
        self.planner = None  # Add Planning

        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """Run the agent on a task"""
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """Perform a single step"""
        messages = [{"role": "user", "content": task}]
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str | None:
        """Execute a tool action."""
        # Placeholder - will be implemented in later chapters
        return f"Executed action: {action}"


# Prepare request
data = json.dumps({
    "model": "gemma4:e4b",
    "messages": [{"role": "user", "content": "Hi!, How's life?"}]
}).encode("utf-8")

# Post request
req = urllib.request.Request(
    url="http://localhost:11434/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

# Parse response
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())

# Print response
print(result["choices"][0])

# Re-initialize the LLM with the updated class
llm = LLM(model="gemma4:e4b")

# Generate a `Response` dataclass
response = llm.generate([{"role": "user", "content": "Hi! How's life?"}])
print(response)

agent = TinyAgent(llm=llm)
response = agent.run("What is 2 + 2?")
print(response)
print(agent.trajectory.runs)
