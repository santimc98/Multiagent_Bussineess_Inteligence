
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch
from src.agents.strategist import StrategistAgent
from src.graph.graph import run_strategist

class TestStrategistNormalization:
    
    def setup_method(self):
        self.agent = StrategistAgent(api_key="dummy_key")

    def test_normalize_dict_with_strategies_list(self):
        """Case: parsed = {'strategies': [{'title': 'A'}, {'title': 'B'}]}"""
        parsed = {"strategies": [{"title": "A"}, {"title": "B"}]}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 2
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_dict_with_strategies_dict(self):
        """Case: parsed = {'strategies': {'title': 'A'}} -> convert to list"""
        parsed = {"strategies": {"title": "A"}}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized["strategies"], list)
        assert len(normalized["strategies"]) == 1
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_list_of_dicts(self):
        """Case: parsed = [{'title': 'A'}, {'title': 'B'}] -> wrap in dict"""
        parsed = [{"title": "A"}, {"title": "B"}]
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 2

    def test_normalize_single_strategy_dict_without_key(self):
        """Case: parsed = {'title': 'A', ...} -> wrap in strategies list"""
        parsed = {"title": "A", "objective_type": "descriptive"}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 1
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_garbage(self):
        """Case: parsed = 'garbage' or None -> empty strategies"""
        assert self.agent._normalize_strategist_output("garbage")["strategies"] == []
        assert self.agent._normalize_strategist_output(None)["strategies"] == []
        assert self.agent._normalize_strategist_output(123)["strategies"] == []


class TestGraphStrategistIntegration:
    
    @patch("src.graph.graph.strategist")
    def test_run_strategist_handles_list_return_legacy(self, mock_strategist):
        """
        Simulate strategist.generate_strategies returning a LIST (legacy bug).
        run_strategist must catch it and form a valid state update.
        """
        # Simulate legacy behavior: returning a list directly
        mock_output = [{"title": "Legacy Strategy", "objective_type": "predictive"}]
        mock_strategist.generate_strategies.return_value = mock_output
        
        # Minimal state
        state = {
            "business_objective": "Test Goal",
            "data_summary": "Some data",
            "run_id": "test_run"
        }
        
        # Mock other dependencies if needed (none strictly needed for this logic branch)
        
        result = run_strategist(state)
        
        # Assertions
        assert "strategies" in result
        strategies_wrapper = result["strategies"]
        assert "strategies" in strategies_wrapper
        strategies_list = strategies_wrapper["strategies"]
        
        assert isinstance(strategies_list, list)
        assert len(strategies_list) == 1
        assert strategies_list[0]["title"] == "Legacy Strategy"
        # Strategy spec falls back to empty dict if not dict result
        assert result.get("strategy_spec") == {}

    @patch("src.graph.graph.strategist")
    def test_run_strategist_handles_dict_return(self, mock_strategist):
        """
        Simulate normal V2 behavior: returning a DICT.
        """
        mock_output = {
            "strategies": [{"title": "Modern Strategy"}],
            "strategy_spec": {"spec": "details"}
        }
        mock_strategist.generate_strategies.return_value = mock_output
        
        state = {"business_objective": "Test", "run_id": "test_run"}
        result = run_strategist(state)
        
        assert result["strategies"]["strategies"][0]["title"] == "Modern Strategy"
        assert result["strategy_spec"]["spec"] == "details"
