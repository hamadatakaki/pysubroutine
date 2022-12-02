# Kurage's Python subroutine repository

色々な研究や課題でよく使いそうなスクリプトをまとめておくめのリポジトリ。

submoduleとしてaddしてlocalで `pip install` ができるようにしました。

## how to install / uninstall

installするときは次のように `git submodule add` と `pip install` を行う。

```sh
$ INSTALL_PATH=./pysubroutine
$ git submodule add git@github.com:hamadatakaki/pysubroutine.git $INSTALL_PATH
$ pip install $INSTALL_PATH
```

uninstallするときは `kurage-subroutine` という名前でできる。

```sh
$ pip uninstall kurage-subroutine
```

## 各種開発ツール

```sh
# test all
$ python -m unittest discover tests
# mypy checking
$ mypy subroutine
# isort
$ isort subroutine
```

## 動作環境

Python3.9 on Ubuntu on WSL2で開発・動作確認をしています。
