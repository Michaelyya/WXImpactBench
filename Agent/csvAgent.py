from langchain.agents import AgentType

from langchain.chat_models import ChatOpenAI
# from langchain.tools import PythonPERLTool
from langchain_experimental.agents import create_csv_agent
from pathlib import Path

def main():
    # print("hi")
    # python_agent_executor = create_python_agent(
    #     llm=ChatOpenAI(temperature=0,model="gpt-4"),
    #     tool=PythonPERLTool(),
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )

    csv_agent= create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="/Users/yonganyu/Desktop/GEOG_research/episode_info.csv",
        verbose=True,
        agent_Type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.run("how many colunbs in this csv file")


if __name__=="__main__":
    
    main()