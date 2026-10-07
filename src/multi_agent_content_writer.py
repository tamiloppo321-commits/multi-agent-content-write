import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


class AgentState(TypedDict):
    user_input: str
    route: str
    output: str
    messages: Annotated[list, add_messages]


@tool
def research_tool(topic: str) -> str:
    """Find deep articles, references, papers and background information."""
    return (
        f"Research brief for '{topic}': key themes, current approaches, "
        "implementation considerations, trade-offs and useful references."
    )


@tool
def internet_search_tool(query: str) -> str:
    """Find SEO keywords, trending topics and hashtags."""
    return (
        f"Search results for '{query}': suggested keywords, trends and "
        "hashtags including #AI #GenAI #AgenticAI."
    )


def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def router_node(state: AgentState):
    text = state["user_input"].lower()

    if any(x in text for x in ["tweet", "twitter", "x post", "280 characters"]):
        route = "x_blog_writer"
    elif any(x in text for x in ["blog", "article", "seo", "long-form", "long form"]):
        route = "seo_blog_writer"
    else:
        route = "general"

    # Optional LLM router when an API key is configured.
    if os.getenv("OPENAI_API_KEY"):
        try:
            prompt = (
                "Classify the request into exactly one label: "
                "seo_blog_writer, x_blog_writer, or general. "
                "Return only the label.\n\n" + state["user_input"]
            )
            candidate = get_llm().invoke([HumanMessage(content=prompt)]).content.strip()
            if candidate in {"seo_blog_writer", "x_blog_writer", "general"}:
                route = candidate
        except Exception:
            pass

    return {
        "route": route,
        "messages": [HumanMessage(content=state["user_input"])],
    }


def seo_blog_writer_node(state: AgentState):
    if not os.getenv("OPENAI_API_KEY"):
        output = (
            f"# {state['user_input']}\n\n"
            "## Introduction\nA practical, search-friendly overview.\n\n"
            "## Key Insights\n- Audience needs\n- Current approaches\n- Implementation considerations\n\n"
            "## Best Practices\nUse evidence, examples and measurable outcomes.\n\n"
            "## Conclusion\nTurn the insights into an actionable next step.\n\n"
            "### CTA\nExplore the next implementation step."
        )
    else:
        messages = [
            SystemMessage(
                content=(
                    "You are an expert SEO blog writer. Use research_tool for "
                    "deep research and internet_search_tool for SEO keywords/trends. "
                    "Write a structured long-form article with headings and CTA."
                )
            ),
            *state["messages"],
        ]
        response = get_llm().bind_tools(
            [research_tool, internet_search_tool]
        ).invoke(messages)
        output = response.content

    return {"output": output, "messages": [AIMessage(content=output)]}


def x_blog_writer_node(state: AgentState):
    if not os.getenv("OPENAI_API_KEY"):
        output = (
            f"🚀 {state['user_input'][:180]}\n\n"
            "AI is moving from demos to practical workflows. "
            "#AI #GenAI #AgenticAI"
        )[:280]
    else:
        response = get_llm().bind_tools([internet_search_tool]).invoke([
            SystemMessage(
                content=(
                    "You are an X/Twitter writer. Write under 280 characters, "
                    "using emojis and relevant hashtags."
                )
            ),
            *state["messages"],
        ])
        output = response.content[:280]

    return {"output": output, "messages": [AIMessage(content=output)]}


def general_node(state: AgentState):
    if not os.getenv("OPENAI_API_KEY"):
        output = f"I can help with your request: {state['user_input']}"
    else:
        response = get_llm().invoke([
            SystemMessage(
                content="You are a helpful general assistant. Use conversation history."
            ),
            *state["messages"],
        ])
        output = response.content

    return {"output": output, "messages": [AIMessage(content=output)]}


def route_after_router(state: AgentState):
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("seo_blog_writer", seo_blog_writer_node)
    graph.add_node("x_blog_writer", x_blog_writer_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "seo_blog_writer": "seo_blog_writer",
            "x_blog_writer": "x_blog_writer",
            "general": "general",
        },
    )

    graph.add_edge("seo_blog_writer", END)
    graph.add_edge("x_blog_writer", END)
    graph.add_edge("general", END)

    return graph.compile(checkpointer=MemorySaver())


APP = build_graph()


def run_agent(user_input: str, thread_id: str = "default"):
    return APP.invoke(
        {
            "user_input": user_input,
            "route": "",
            "output": "",
            "messages": [],
        },
        config={"configurable": {"thread_id": thread_id}},
    )


if __name__ == "__main__":
    examples = [
        "Write an SEO blog about enterprise AI",
        "Create an X post about RAG",
        "Hello, what can you help me with?",
    ]

    for query in examples:
        result = run_agent(query)
        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("ROUTE:", result["route"])
        print("OUTPUT:\n", result["output"])
