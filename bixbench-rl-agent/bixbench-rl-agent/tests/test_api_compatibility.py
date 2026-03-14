"""
验证本项目代码与真实 aviary/ldp API 兼容。
"""
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
ROOT = Path(__file__).parent.parent


def test_notebook_env_inherits_nbenv():
    source = (SRC / "envs/notebook_env.py").read_text()
    assert "class ControlledNotebookEnv(NBEnvironment" in source

def test_uses_hide_old_env_states():
    source = (SRC / "agents/notebook_agent.py").read_text()
    assert "hide_old_env_states=True" in source

def test_uses_real_react_agent():
    source = (SRC / "agents/notebook_agent.py").read_text()
    assert "from ldp.agent import ReActAgent" in source

def test_step_grpo_groups_by_step():
    source = (ROOT / "scripts/train_grpo.py").read_text()
    assert "by_step" in source
    assert "step_num" in source

def test_grpo_config_step_level():
    cfg = (ROOT / "configs/grpo_step.yaml").read_text()
    assert "group_size: 1" in cfg
    assert "use_transitions: true" in cfg
    assert "step_level" in cfg

def test_env_adds_submit_answer():
    source = (SRC / "envs/notebook_env.py").read_text()
    assert "submit_answer" in source

def test_setup_uses_real_repos():
    setup = (ROOT / "scripts/setup_env.sh").read_text()
    assert "Future-House/aviary" in setup
    assert "Future-House/BixBench" in setup
    assert "NVIDIA-NeMo/Gym" in setup
    assert "NVIDIA-NeMo/RL" in setup

def test_uses_official_docker_image():
    setup = (ROOT / "scripts/setup_env.sh").read_text()
    assert "futurehouse/bixbench-env:v1.0" in setup

def test_no_mock_kernel_in_new_project():
    for py_file in SRC.rglob("*.py"):
        content = py_file.read_text()
        assert "_MockKernelBackend" not in content, \
            f"{py_file.name} 包含 mock kernel（应使用真实 Docker kernel）"

def test_bixbench_state_extends_nbstate():
    source = (SRC / "envs/notebook_env.py").read_text()
    assert "class BixBenchNBState(NBEnvironmentState" in source
    assert "task_instruction" in source
    assert "ground_truth" in source

def test_context_truncation_via_ldp_not_custom():
    """Edison 创新 #1 应通过 ldp 内置参数实现，不是自定义循环。"""
    source = (SRC / "agents/notebook_agent.py").read_text()
    # 正确方式：ldp 参数
    assert "hide_old_env_states" in source
    # 不应有自定义的 context_manager 模块
    assert "ContextManager" not in source
