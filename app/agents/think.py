from langchain_core.tools import tool


@tool
def think_tool(thought: str) -> str:
    """Остановись и подумай. Запиши: что узнал, что осталось, какой следующий шаг.

    Используй до и после каждого web_search и перед сменой направления.
    """
    return thought
