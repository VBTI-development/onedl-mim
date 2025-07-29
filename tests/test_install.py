# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import tempfile

from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall


def test_third_party():
    runner = CliRunner()
    # mim install fire
    result = runner.invoke(install, ['fire'])
    assert result.exit_code == 0, result.output

    # mim uninstall fire --yes
    result = runner.invoke(uninstall, ['fire', '--yes'])
    assert result.exit_code == 0, result.output


def test_mmcv_install():
    runner = CliRunner()
    # mim install onedl-mmcv --yes
    # install latest version
    result = runner.invoke(install, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output

    # mim install onedl-mmcv==1.3.1 --yes
    result = runner.invoke(install, ['onedl-mmcv==1.3.1', '--yes'])
    assert result.exit_code == 0, result.output

    # mim uninstall onedl-mmcv --yes
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output

    # version should be less than latest version
    # mim install onedl-mmcv==100.0.0 --yes
    result = runner.invoke(install, ['onedl-mmcv==100.0.0', '--yes'])
    assert result.exit_code == 1


def test_mmrepo_install():
    runner = CliRunner()

    # install local repo
    with tempfile.TemporaryDirectory() as temp_root:
        repo_root = osp.join(temp_root, 'mmclassification')
        subprocess.check_call([
            'git', 'clone',
            'https://github.com/vbti-development/onedl-mmpretrain.git',
            repo_root
        ])

        # mim install .
        current_root = os.getcwd()
        os.chdir(repo_root)
        result = runner.invoke(install, ['.', '--yes'])
        assert result.exit_code == 0, result.output

        os.chdir('..')

        # mim install ./mmclassification
        result = runner.invoke(install, ['./mmclassification', '--yes'])
        assert result.exit_code == 0, result.output

        # mim install -e ./mmclassification
        result = runner.invoke(install, ['-e', './mmclassification', '--yes'])
        assert result.exit_code == 0, result.output

        os.chdir(current_root)

    # mim install git+https://github.com/vbti-development/onedl-mmpretrain.git
    result = runner.invoke(
        install,
        ['git+https://github.com/vbti-development/onedl-mmpretrain.git'])
    assert result.exit_code == 0, result.output

    # mim install onedl-mmpretrain --yes
    result = runner.invoke(install, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output

    # mim install onedl-mmpretrain==0.11.0 --yes
    result = runner.invoke(install, ['onedl-mmpretrain==0.11.0', '--yes'])
    assert result.exit_code == 0, result.output

    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output

    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
