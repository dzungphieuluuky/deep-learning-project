import pytest
import torch
from omegaconf import OmegaConf
from project_name.models.transformer import TransformerModel


@pytest.fixture
def model_cfg():
    return OmegaConf.create({
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "max_seq_len": 128,
        "vocab_size": 1000,
        "use_flash_attention": False,
    })


@pytest.fixture
def model(model_cfg):
    return TransformerModel(model_cfg)


def test_model_forward(model):
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": None,
    }
    outputs = model(batch)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 32, 1000)


def test_model_loss(model):
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": None,
    }
    outputs = model(batch)
    loss_dict = model.compute_loss(outputs, batch)
    assert "loss" in loss_dict
    assert loss_dict["loss"].ndim == 0   # scalar
    assert not torch.isnan(loss_dict["loss"])


def test_model_parameter_count(model):
    params = model.count_parameters()
    assert params["total"] > 0
    assert params["trainable"] <= params["total"]


def test_model_device_transfer(model):
    if torch.cuda.is_available():
        model = model.cuda()
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 32)).cuda(),
            "attention_mask": None,
        }
        outputs = model(batch)
        assert outputs["logits"].device.type == "cuda"
