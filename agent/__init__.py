from agent.agents import Agent3x


def get_training_agent(config, net):
    return Agent3x(config, net)
