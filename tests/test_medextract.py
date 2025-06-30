"""
Unit tests for medextract.py
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import yaml
import sys
import os

# Add the parent directory to sys.path to import medextract
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import medextract


class TestMedExtract(unittest.TestCase):
    """Test cases for medextract module"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_config = {
            'evaluation': {
                'target_variable': 'BTFU Score (Updated)',
                'valid_values': ['0', '1', '1a', '1b', '2', '2a', '2b', '3', '3a', '3b', '3c', '4', 'NR']
            },
            'system_prompts': {
                'simple': 'Extract the {target_variable} from the given medical report.',
                'complex': 'You are an AI assistant specialized in extracting {target_variable}.'
            },
            'few_shot_examples': {
                'example1': {
                    'input': 'Sample input',
                    'output': 'Sample output'
                }
            }
        }

    def test_preprocess_text(self):
        """Test text preprocessing function"""
        input_text = "Line 1\nLine 2.\nLine 3"
        expected = "Line 1 Line 2. \n Line 3"
        result = medextract.preprocess_text(input_text)
        self.assertEqual(result, expected)

    def test_clean_extracted_value_valid(self):
        """Test clean_extracted_value with valid JSON"""
        with patch.dict('medextract.config', self.sample_config):
            # Test valid JSON string
            json_string = '{"BTFU Score (Updated)": "1a"}'
            result = medextract.clean_extracted_value(json_string)
            self.assertEqual(result, "1a")
            
            # Test valid dict
            json_dict = {"BTFU Score (Updated)": "2"}
            result = medextract.clean_extracted_value(json_dict)
            self.assertEqual(result, "2")

    def test_clean_extracted_value_invalid(self):
        """Test clean_extracted_value with invalid values"""
        with patch.dict('medextract.config', self.sample_config):
            # Test invalid value
            json_string = '{"BTFU Score (Updated)": "invalid_score"}'
            result = medextract.clean_extracted_value(json_string)
            self.assertEqual(result, "invalid")
            
            # Test malformed JSON
            invalid_json = "not valid json"
            result = medextract.clean_extracted_value(invalid_json)
            self.assertEqual(result, "invalid")

    @patch('medextract.yaml.safe_load')
    @patch('builtins.open')
    def test_load_config_default(self, mock_open, mock_yaml_load):
        """Test config loading with default config"""
        # Mock file reading
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        # Mock default config
        default_config = {'use_default_config': True, 'test': 'default'}
        user_config = {'use_default_config': True}
        
        # Mock yaml loading to return different configs
        mock_yaml_load.side_effect = [default_config, user_config]
        
        result = medextract.load_config()
        
        # Should return default config when use_default_config is True
        self.assertEqual(result, default_config)

    @patch('medextract.yaml.safe_load')
    @patch('builtins.open')
    def test_load_config_custom(self, mock_open, mock_yaml_load):
        """Test config loading with custom config"""
        # Mock file reading
        mock_open.return_value.__enter__.return_value = MagicMock()
        
        # Mock configs
        default_config = {'test': 'default', 'default_only': 'value'}
        user_config = {'use_default_config': False, 'test': 'custom'}
        expected_merged = {'test': 'custom', 'default_only': 'value', 'use_default_config': False}
        
        # Mock yaml loading
        mock_yaml_load.side_effect = [default_config, user_config]
        
        result = medextract.load_config()
        
        # Should return merged config when use_default_config is False
        self.assertEqual(result, expected_merged)

    def test_construct_prompt_simple(self):
        """Test prompt construction with simple prompting"""
        with patch.dict('medextract.config', self.sample_config):
            context = "Patient has score 2a"
            result = medextract.construct_prompt(
                context, 
                simple_prompting=True, 
                fewshots_method=False, 
                fewshots_with_NR_method=False, 
                fewshots_with_NR_extended_method=False
            )
            
            # Check that system message is formatted correctly
            self.assertEqual(len(result), 2)  # system + user message
            self.assertIn("BTFU Score (Updated)", result[0]['content'])
            self.assertIn(context, result[1]['content'])


class TestBgeRerank(unittest.TestCase):
    """Test cases for BgeRerank class"""

    @patch('medextract.CrossEncoder')
    def test_bge_rerank_initialization(self, mock_cross_encoder):
        """Test BgeRerank initialization"""
        reranker = medextract.BgeRerank()
        self.assertEqual(reranker.model_name, 'BAAI/bge-reranker-v2-m3')
        self.assertEqual(reranker.top_n, 2)

    @patch('medextract.CrossEncoder')
    def test_bge_rerank_method(self, mock_cross_encoder):
        """Test bge_rerank method"""
        # Mock the CrossEncoder
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.3]
        mock_cross_encoder.return_value = mock_model
        
        reranker = medextract.BgeRerank()
        query = "test query"
        docs = ["doc1", "doc2", "doc3"]
        
        result = reranker.bge_rerank(query, docs)
        
        # Should return top 2 results sorted by score
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0)  # Index of highest score
        self.assertEqual(result[1][0], 1)  # Index of second highest score


if __name__ == '__main__':
    unittest.main()