# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path


def get_version() -> str:
    root_dir = os.path.dirname(__file__)
    version_file = os.path.join(root_dir, "sfl", "version.py")
    with open(version_file, "rt", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'^__version__\s*=\s*["\'](.*?)["\']', content, re.M)
    if not m:
        raise RuntimeError("Cannot find __version__ in sfl/version.py")
    version = m.group(1)

    if "$$DATE$$" in version:
        date_str = date.today().strftime("%Y%m%d")
        return version.replace("$$DATE$$", date_str)

    return version


def get_commit_id() -> str:
    commit_id = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    dirty = subprocess.check_output(["git", "diff", "--stat"]).decode("ascii").strip()

    if dirty:
        commit_id = f"{commit_id}-dirty"

    return commit_id


def update_version_file(filepath):
    # Proceed with file modifications
    today = date.today()
    dstr = today.strftime("%Y%m%d")
    with open(filepath, "r") as fp:
        content = fp.read()

    content = content.replace("$$DATE$$", dstr)
    content = content.replace("$$BUILD_TIME$$", time.strftime("%b %d %Y, %X"))
    try:
        content = content.replace("$$COMMIT_ID$$", get_commit_id())
    except:
        pass

    if "SF_BUILD_DOCKER_NAME" in os.environ:
        content = content.replace(
            "$$DOCKER_VERSION$$", os.environ["SF_BUILD_DOCKER_NAME"]
        )

    with open(filepath, "w+") as fp:
        fp.write(content)


VERSION_FILE = Path("sfl/version.py")
BACKUP_FILE = Path("sfl/version.py.bak")


def pdm_build_initialize(context):
    # pdm hook function, please refer to https://backend.pdm-project.org/hooks/
    # backup version file,and update version
    if BACKUP_FILE.exists():
        raise ValueError(f"{BACKUP_FILE} exists, please check it.")

    shutil.copy2(VERSION_FILE, BACKUP_FILE)
    update_version_file(VERSION_FILE)

    def restore():
        if BACKUP_FILE.exists():
            shutil.copy2(BACKUP_FILE, VERSION_FILE)
            BACKUP_FILE.unlink()

    atexit.register(restore)


def _build_lib(force: bool = False) -> Path:
    target = f"//libsfl/binding:binding"

    dst_path = Path("sfl/security/privacy/_lib.so")

    if dst_path.exists():
        if force:
            dst_path.unlink()
        else:
            print(f"{dst_path} exists, ignore build")
            return dst_path

    ver_info = sys.version_info
    version = f"{ver_info.major}.{ver_info.minor}"
    args = [
        "bazel",
        "build",
        target,
        f"--@rules_python//python/config_settings:python_version={version}",
    ]
    subprocess.run(args, check=True)
    so_path = Path("bazel-bin/libsfl/binding/_lib.so")
    shutil.copy2(so_path, dst_path)
    return dst_path


def bazel_clean():
    # pdm script command, please see the config([tool.pdm.scripts]) in pyproject.toml.
    so_path = Path("sfl/security/privacy/_lib.so")
    so_path.unlink(missing_ok=True)


def bazel_build():
    # pdm script command, please see the config([tool.pdm.scripts]) in pyproject.toml.
    # build so using bazel and copy to sfl
    force = os.getenv("SFL_BUILD_FORCE", "false") == "true"
    _build_lib(force)


def pdm_build_update_files(context, files: dict):
    # pdm hook function, please refer to https://backend.pdm-project.org/hooks/
    # When 'python -m build' is executed, this function will be auto called
    # and the *.so built by bazel will be packaged into sfl*.whl
    force = os.getenv("SFL_BUILD_FORCE", "false") == "true"
    libsfl = _build_lib(force)
    files[str(libsfl)] = libsfl
