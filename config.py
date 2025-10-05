# config.py
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    db_type: str = "sqlite"
    host: str = "localhost"
    database: str = "log_analysis"
    user: str = "postgres"
    password: str = ""
    port: str = "5432"
    sqlite_path: str = "log_analysis.db"

@dataclass
class ModelConfig:
    name: str = "llama3:latest"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

@dataclass
class AppConfig:
    vectorstore_dir: str = "."
    index_name: str = "faiss_index"
    default_analysis_days: int = 7
    recent_logs_limit: int = 100

class ConfigManager:
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.model_config = ModelConfig()
        self.app_config = AppConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for session state"""
        return {
            "db_config": self.db_config.__dict__,
            "model_config": self.model_config.__dict__,
            "app_config": self.app_config.__dict__
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load config from dictionary"""
        if "db_config" in config_dict:
            for key, value in config_dict["db_config"].items():
                setattr(self.db_config, key, value)
        if "model_config" in config_dict:
            for key, value in config_dict["model_config"].items():
                setattr(self.model_config, key, value)
        if "app_config" in config_dict:
            for key, value in config_dict["app_config"].items():
                setattr(self.app_config, key, value)