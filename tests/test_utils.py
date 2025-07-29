# Copyright (c) OpenMMLab. All rights reserved.
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall
from mim.utils import get_github_url, parse_home_page
from mim.utils.utils import get_torch_device_version, is_npu_available


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output


def test_parse_home_page():
    runner = CliRunner()
    result = runner.invoke(install, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
    assert parse_home_page(
        'onedl-mmpretrain'
    ) == 'https://github.com/vbti-development/onedl-mmpretrain'
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output


def test_get_github_url():
    runner = CliRunner()
    result = runner.invoke(install, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
    assert get_github_url(
        'onedl-mmpretrain'
    ) == 'https://github.com/vbti-development/onedl-mmpretrain.git'

    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
    assert get_github_url(
        'onedl-mmpretrain'
    ) == 'https://github.com/vbti-development/onedl-mmpretrain.git'


def test_get_torch_device_version():
    torch_v, device, device_v = get_torch_device_version()
    assert torch_v.replace('.', '').isdigit()
    if is_npu_available():
        assert device == 'ascend'


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
