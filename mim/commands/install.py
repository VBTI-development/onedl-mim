# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from importlib import metadata as importlib_metadata
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import click

from mim.utils import (
    DEFAULT_MMCV_BASE_URL,
    PKG2PROJECT,
    call_command,
    echo_warning,
    get_torch_device_version,
)


@click.command(
    'install',
    context_settings=dict(ignore_unknown_options=True),
)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@click.option(
    '-i',
    '--index-url',
    '--pypi-url',
    'index_url',
    help='Base URL of the Python Package Index (default %default). '
    'This should point to a repository compliant with PEP 503 '
    '(the simple repository API) or a local directory laid out '
    'in the same format.')
@click.option(
    '-y',
    '--yes',
    'is_yes',
    is_flag=True,
    help="Don't ask for confirmation of uninstall deletions. "
    'Deprecated, will have no effect.')
def cli(
    args: Tuple[str],
    index_url: Optional[str] = None,
    is_yes: bool = False,
) -> None:
    """Install packages.

    You can use `mim install` in the same way you use `pip install`!

    And `mim install` will **install the 'mim' extra requirements**
    for OneDL Lab packages if needed.

    \b
    Example:
        > mim install mmdet onedl-mmpretrain
        > mim install git+https://github.com/vbti-development/onedl-mmdetection.git  # noqa: E501
        > mim install -r requirements.txt
        > mim install -e <path>
        > mim install mmdet -i <url> -f <url>
        > mim install mmdet --extra-index-url <url> --trusted-host <hostname>

    Here we list several commonly used options.
    For more options, please refer to `pip install --help`.
    """
    if is_yes:
        echo_warning(
            'The `--yes` option has been deprecated, will have no effect.')
    exit_code = install(list(args), index_url=index_url)
    exit(exit_code)


def install(
    install_args: List[str],
    index_url: Optional[str] = None,
) -> Any:
    """Install packages via pip and add 'mminstall' extra requirements for
    OneDL Lab packages during pip install process.

    Args:
        install_args (list): List of arguments passed to `pip install`.
        index_url (str, optional): The pypi index url.

    Returns:
        int: Exit code from pip install.
    """
    # Modify install args to add [mminstall] extras for OneDL packages
    modified_args = add_mminstall_extras(install_args)

    # Add mmcv find-links
    modified_args = add_mmcv_find_links(modified_args)

    # Add index URL if provided
    if index_url is not None:
        modified_args += ['-i', index_url]

    # Run pip install using subprocess
    pip_cmd = [sys.executable, '-m', 'pip', 'install'] + modified_args
    result = call_command(pip_cmd)

    check_mim_resources()
    return result


def add_mminstall_extras(install_args: Sequence[str]) -> List[str]:
    """Add [mminstall] extras to OneDL Lab packages."""
    modified_args = []

    for arg in install_args:
        # Skip option flags
        if arg.startswith('-'):
            modified_args.append(arg)
            continue

        # Check if this is an OneDL Lab package
        package_name = arg.split('[')[0].split('=')[0].split('<')[0].split(
            '>')[0].split('!')[0].split('~')[0].strip()

        if package_name in PKG2PROJECT and package_name != 'onedl-mmcv':
            # Add [mminstall] if not already present and no other extras
            if '[' not in arg:
                # Simple case: package_name -> package_name[mminstall]
                rest_of_spec = arg[len(package_name):]
                modified_arg = f'{package_name}[mminstall]{rest_of_spec}'
                modified_args.append(modified_arg)
            elif 'mminstall' not in arg:
                # Has extras but not mminstall:
                # package[extra] -> package[extra,mminstall]
                bracket_pos = arg.find('[')
                close_bracket_pos = arg.find(']')
                if bracket_pos != -1 and close_bracket_pos != -1:
                    before_bracket = arg[:bracket_pos]
                    extras = arg[bracket_pos + 1:close_bracket_pos]
                    after_bracket = arg[close_bracket_pos + 1:]
                    modified_arg = \
                        f'{before_bracket}[{extras},mminstall]{after_bracket}'
                    modified_args.append(modified_arg)
                else:
                    modified_args.append(arg)
            else:
                # Already has mminstall
                modified_args.append(arg)
        else:
            modified_args.append(arg)

    return modified_args


def add_mmcv_find_links(install_args: List[str]) -> List[str]:
    """Add mmcv find-links to install arguments."""
    # Get mmcv_base_url from environment variable if exists
    mmcv_base_url = os.environ.get('MMCV_BASE_URL', DEFAULT_MMCV_BASE_URL)

    if mmcv_base_url != DEFAULT_MMCV_BASE_URL:
        echo_warning('Using the mmcv find base URL from environment variable '
                     f'`MMCV_BASE_URL`: {mmcv_base_url}')

    # Check URL format
    parse_result = urlparse(mmcv_base_url)
    if not parse_result.scheme:
        echo_warning(f'Invalid MMCV_BASE_URL: {mmcv_base_url}. Using default.')
        mmcv_base_url = DEFAULT_MMCV_BASE_URL
        parse_result = urlparse(mmcv_base_url)

    modified_args = install_args.copy()

    # Mark mmcv find host as trusted if URL scheme is http
    if parse_result.scheme == 'http':
        modified_args += ['--trusted-host', parse_result.netloc]

    # Add onedl-mmcv find links
    find_link = get_mmcv_full_find_link(mmcv_base_url)
    modified_args += ['-f', find_link]

    return modified_args


def get_mmcv_full_find_link(mmcv_base_url: str) -> str:
    """Get the onedl-mmcv find link corresponding to the current
    environment."""
    torch_v, device, device_v = get_torch_device_version()
    major, minor, *_ = torch_v.split('.')
    torch_v = '.'.join([major, minor, '0'])

    if device == 'cuda' and device_v.isdigit():
        device_link = f'cu{device_v}'
    elif device == 'ascend':
        device_link = f'ascend{device_v}'
    else:
        device_link = 'cpu'

    find_link = f'{mmcv_base_url}/{device_link.replace(".", "")}-torch{torch_v.replace(".", "")}/simple/onedl-mmcv/index.html'  # noqa: E501
    return find_link


def check_mim_resources() -> None:
    """Check if the mim resource directory exists."""
    for dist in importlib_metadata.distributions():
        try:
            pkg_name = dist.name
            if pkg_name is None:
                continue
        except (OSError, AttributeError):
            # Skip corrupted distributions
            continue

        normalized_pkg_name = pkg_name.lower().replace('_', '-')

        if (normalized_pkg_name not in PKG2PROJECT
                or normalized_pkg_name == 'onedl-mmcv'):
            continue

        # Find the installed package location
        installed_path = None

        try:
            # Method 1: Use top_level.txt
            top_level = dist.read_text('top_level.txt')
            if top_level:
                module_name = top_level.split('\n')[0].strip()
                potential_path = os.path.join(
                    str(dist.locate_file('.')), module_name)
                if os.path.exists(potential_path):
                    installed_path = potential_path
        except (FileNotFoundError, AttributeError, OSError):
            pass

        if not installed_path:
            try:
                # Method 2: Try the package name as module name
                potential_path = os.path.join(
                    str(dist.locate_file('.')), pkg_name.replace('-', '_'))
                if os.path.exists(potential_path):
                    installed_path = potential_path
            except (OSError, AttributeError):
                pass

        if not installed_path:
            try:
                # Method 3: Use the distribution location directly
                installed_path = str(dist.locate_file('.'))
            except (OSError, AttributeError):
                echo_warning(f'Cannot locate files for {pkg_name}, skipping')
                continue

        mim_resources_path = os.path.join(installed_path, '.mim')
        if not os.path.exists(mim_resources_path):
            echo_warning(f'mim resources not found: {mim_resources_path}, '
                         f'you can try to install the latest {pkg_name}.')
