def test_imports_are_local_only():
    import agent
    import harness
    assert callable(harness.SeaVideoHarness)
    assert callable(agent.AgentController)
