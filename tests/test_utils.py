# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from mim.commands.install import cli as install
from mim.commands.uninstall import cli as uninstall
from mim.utils import get_github_url, parse_home_page
from mim.utils.utils import (
    get_installed_path,
    get_torch_device_version,
    is_npu_available,
)


def setup_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output


def test_parse_home_page():
    runner = CliRunner()
    result = runner.invoke(install, ['onedl-mmengine', '--yes'])
    assert result.exit_code == 0, result.output
    assert parse_home_page(
        'onedl-mmengine'
    ) == 'https://github.com/vbti-development/onedl-mmengine'
    result = runner.invoke(uninstall, ['onedl-mmengine', '--yes'])
    assert result.exit_code == 0, result.output


def test_get_github_url():
    runner = CliRunner()
    result = runner.invoke(install, ['onedl-mmengine', '--yes'])
    assert result.exit_code == 0, result.output
    assert get_github_url(
        'onedl-mmengine'
    ) == 'https://github.com/vbti-development/onedl-mmengine.git'

    result = runner.invoke(uninstall, ['onedl-mmengine', '--yes'])
    assert result.exit_code == 0, result.output
    assert get_github_url(
        'onedl-mmengine'
    ) == 'https://github.com/vbti-development/onedl-mmengine.git'


def test_get_torch_device_version():
    torch_v, device, device_v = get_torch_device_version()
    assert torch_v.replace('.', '').isdigit()
    if is_npu_available():
        assert device == 'ascend'


def _make_editable_dist(url):
    """Return a mock Distribution whose direct_url.json holds *url*."""
    mock_dist = MagicMock()
    mock_dist.read_text.return_value = json.dumps({
        'url': url,
        'dir_info': {
            'editable': True
        },
    })
    return mock_dist


@pytest.mark.skipif(os.name != 'nt', reason='Windows-specific path handling')
def test_get_installed_path_windows_strips_leading_slash(tmp_path):
    """On Windows, urlparse('file:///D:/path').path starts with '/D:/', which
    normpath turns into '\\D:\\' — an invalid path. The code must strip that
    leading slash so the result starts with the drive letter."""
    real_dir = tmp_path / 'mm-project' / 'onedl-mmdetection'
    real_dir.mkdir(parents=True)
    mock_dist = _make_editable_dist(real_dir.as_uri())

    with patch('mim.utils.utils.is_installed', return_value=True), \
         patch('mim.utils.utils.importlib_metadata.distribution',
               return_value=mock_dist):
        result = get_installed_path('onedl-mmdetection')

    assert not result.startswith('\\'), (
        f'Leading backslash before drive letter: {result!r}')
    assert result[1:3] == ':\\', (
        f'Expected drive letter at start, got: {result!r}')


@pytest.mark.skipif(os.name != 'nt', reason='Windows-specific path handling')
def test_get_installed_path_windows_restores_casing(tmp_path):
    """When pip writes a lowercase path to direct_url.json (a known pip/pathlib
    bug on Windows), get_installed_path should recover the canonical casing by
    calling os.path.realpath, which uses GetFinalPathNameByHandle on
    Windows."""
    real_dir = tmp_path / 'MM-Project' / 'onedl-mmdetection'
    real_dir.mkdir(parents=True)
    # Simulate what pip does:
    # store the path as all-lowercase in direct_url.json
    lowercased_url = real_dir.as_uri().lower()
    mock_dist = _make_editable_dist(lowercased_url)

    with patch('mim.utils.utils.is_installed', return_value=True), \
         patch('mim.utils.utils.importlib_metadata.distribution',
               return_value=mock_dist):
        result = get_installed_path('onedl-mmdetection')

    # os.path.realpath should have restored the canonical on-disk casing
    assert result == os.path.realpath(str(real_dir)), (
        f'Expected canonical casing {os.path.realpath(str(real_dir))!r}, '
        f'got {result!r}')


def teardown_module():
    runner = CliRunner()
    result = runner.invoke(uninstall, ['onedl-mmcv', '--yes'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(uninstall, ['onedl-mmpretrain', '--yes'])
    assert result.exit_code == 0, result.output
