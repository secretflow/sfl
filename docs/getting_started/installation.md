# Installation

## Environment

Python：3.10

pip: >= 19.3

OS: CentOS 7, Ubuntu 20.04

CPU/Memory: recommended minimum requirement is 8C16G.

> **_Note:_** Due to CI resource limitation, macOS x64 prebuild binary is no longer available.

## Option 1: from pypi

For users who want to try SecretFlow, you can install [the current release](https://pypi.org/project/secretflow/).

Note that it requires python version == 3.10, you can create a virtual environment with conda if not satisfied.

```
conda create -n sf python=3.10
conda activate sf
```

After that, please use pip to install SecretFlow.

<!-- OPENSOURCE-CLEANUP REMOVE KEYWORD_ONLY -i|https://artifacts.antgroup-inc.cn/simple/| -->

```bash
pip install -U -i https://artifacts.antgroup-inc.cn/simple/ sfl
```

## Option 2: from source

1. Download code and set up Python virtual environment.

```sh
git clone https://github.com/secretflow/sfl.git
cd secretflow

conda create -n secretflow python=3.10
conda activate sfl
```

1. Install SFL

```sh

python -m build --wheel

pip install dist/*.whl
```

## Option 3: from WSL

SecretFlow does not support Windows directly now, however, a Windows user can use secretFlow by WSL(Windows Subsystem for Linux).

1. Install WSL2 in Windows

- You are supposed to follow the [guide_zh](https://learn.microsoft.com/zh-cn/windows/wsl/install) or [guide_en](https://learn.microsoft.com/en-us/windows/wsl/install) to install WSL(Windows Subsystem for Linux) in your Windows and make sure that the version of WSL is 2.
- As for the distribution of GNU/Linux, Ubuntu is recommended.

2. Install Anaconda in WSL

Just follow the installation of anaconda in GNU/Linux to install anaconda in your WSL.

3. Install SFL

- create conda environment

```shell
conda create -n sf python=3.10
```

- activate the environment

```shell
conda activate sf
```

- use pip to install SFL.

```
pip install -U sfl
```

1. Use WSL to develop your application

After set up of SecretFlow in WSL, you can use [Pycharm Professional to Configure an interpreter using WSL](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html) or [Visual Studio Code with WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode) to use SecretFlow in Windows Operating System.

## A quick try

Try your first SecretFlow program.

Import secretflow package.

```python
>>> import secretflow as sf
```

Create a local cluster with parties alice, bob and carol.

```python
>>> sf.init(parties=['alice', 'bob', 'carol'], address='local')
```

Create alice's PYU device, which can process alice's data.

```python
>>> alice_device = sf.PYU('alice')
```

Let alice say hello world.

```python
>>> message_from_alice = alice_device(lambda x:x)("Hello World!")
```

Print the message.

```python
>>> message_from_alice
<secretflow.device.device.pyu.PYUObject object at 0x7fdec24a15b0>
```

We see that the message on alice device is a PYU Object at deriver program.

Print the text at the driver by revealing the message.

```python
>>> print(sf.reveal(message_from_alice))
Hello World!
```
