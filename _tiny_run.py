
from run import main
from config import ModelConfig, RunConfig
mc = ModelConfig()
rc = RunConfig(steps=10, run_name="tiny")
main(mc, rc)
