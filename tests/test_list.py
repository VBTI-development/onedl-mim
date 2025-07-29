# Copyright (c) OpenMMLab. All rights reserved.
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.list import list_package
from mim.commands.uninstall import cli as uninstall


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output


def test_list():
    runner = CliRunner()
    # mim install onedl-mmpretrain==0.23.0 --yes
    result = runner.invoke(install, ['onedl-mmpretrain==0.23.0', '--yes'])
    assert result.exit_code == 0, result.output
    # mim list
    target = ('onedl-mmpretrain', '0.23.0',
              'https://github.com/vbti-development/onedl-mmpretrain.git')
    result = list_package()
    assert target in result


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
