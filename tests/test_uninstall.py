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
    result = runner.invoke(uninstall, ['mmsegmentation', '--yes'])
    assert result.exit_code == 0, result.output


def test_uninstall():
    runner = CliRunner()

    # mim install mmsegmentation --yes
    result = runner.invoke(install, ['mmsegmentation', '--yes'])
    # Use importlib reload module in the same process may cause `isinstance`
    # invalidation.
    # A known issue: `METADATA not found in /tmp/xxx/xxx.whel` will be warning
    # in pip 21.3.1, and onedl-mmcv could not install success as expected.
    # So here we install mmsegmentation twice as an ugly workaround.
    # TODO: find a better way to deal with this issues.
    result = runner.invoke(install, ['mmsegmentation', '--yes'])
    assert result.exit_code == 0, result.output

    # check if install success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmsegmentation' in installed_packages
    assert 'onedl-mmcv' in installed_packages
    # `mim install mmsegmentation` will install mim extra requirements (via
    # mminstall.txt) automatically since PR#132, so we got installed onedl-mmpretrain here.  # noqa: E501
    assert 'onedl-mmpretrain' in installed_packages

    # mim uninstall mmsegmentation --yes
    result = runner.invoke(uninstall, ['mmsegmentation', '--yes'])
    assert result.exit_code == 0, result.output

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'mmsegmentation' not in installed_packages

    # mim uninstall onedl-mmpretrain onedl-mmcv --yes
    result = runner.invoke(uninstall,
                           ['onedl-mmpretrain', 'onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output

    # check if uninstall success
    result = list_package()
    installed_packages = [item[0] for item in result]
    assert 'onedl-mmpretrain' not in installed_packages
    assert 'onedl-mmcv' not in installed_packages


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['mmsegmentation', '--yes'])
    assert result.exit_code == 0, result.output
