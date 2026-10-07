from src.multi_agent_content_writer import router_node


def state(text):
    return {
        "user_input": text,
        "route": "",
        "output": "",
        "messages": [],
    }


def test_seo_route():
    assert router_node(state("Write an SEO blog about AI"))["route"] == "seo_blog_writer"


def test_x_route():
    assert router_node(state("Create a tweet about RAG"))["route"] == "x_blog_writer"


def test_general_route():
    assert router_node(state("Hello"))["route"] == "general"
