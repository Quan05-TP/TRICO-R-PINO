# 协作方式

常规协作使用各自的 Git 分支和 Pull Request；远程开发使用服务器上的独立
checkout；需要结对时再开启 Live Share 会话。这里提供操作说明，没有配置或连接
任何服务器，也没有邀请任何协作者。

## GitHub 权限与分支

每位协作者使用自己的 GitHub 账号。直接向本仓库推送分支需要仓库写权限；账号尚未
指定时，由所有者先确认对方的用户名，再在 GitHub 页面操作：

1. 打开仓库的 **Settings**。
2. 在 **Access** 下打开 **Collaborators**；组织仓库通常显示
   **Collaborators & teams**。
3. 点击 **Add people**，确认用户名后发送邀请；组织仓库按需要选择 **Write**。
4. 由协作者接受邀请后确认权限。个人账号仓库的 collaborator 可写；组织仓库的
   权限由所有者或管理员管理。

这是待所有者执行的步骤，不代表邀请已经发送。参见 GitHub 官方的
[个人仓库邀请说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)
和[组织仓库访问管理](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/managing-teams-and-people-with-access-to-your-repository)。

各自在独立 clone 中从默认分支创建工作分支，通过 PR 提交审查。不要让多人在同一
checkout 中切换分支，也不要直接替换另一个人的未提交文件。结对会话结束后，仍由
该分支的负责人检查 diff、运行检查并提交 PR。具体命令见
[CONTRIBUTING.md](../CONTRIBUTING.md)。

## VS Code 本地开发

1. 用 Python 3.10+ 创建 `.venv`，激活后执行
   `python -m pip install -e ".[dev]"`。
2. 打开 `TRICO-R-PINO.code-workspace`，安装推荐的 Python、Python Debugger 和 Ruff
   扩展，并用 **Python: Select Interpreter** 选择 `.venv`。
3. **Terminal: Run Task** 提供 Doctor、Example、Tests、Lint 和 Format check。
   **Run and Debug** 提供 `pure_pino doctor` 与 `pure_pino example` 的调试入口。
4. 提交前运行 `python -m pytest`、`python -m ruff check .` 和
   `python -m ruff format --check .`，记录实际结果。

## VS Code Remote-SSH

Remote-SSH 在已授权服务器上打开远程目录并运行开发工具。每位协作者使用自己的
服务器访问授权；GitHub 仓库权限和 SSH 登录权限分别管理。

1. 本地安装 **Remote - SSH** 扩展，先确认自己的 SSH 登录可用。
2. 在用户自己的 SSH 配置中管理主机别名、端口和密钥。不要将密码、私钥、真实主机
   地址或个人 SSH 配置提交到本仓库。
3. 执行 **Remote-SSH: Connect to Host...**，选择获准访问的主机。
4. 在远程窗口打开自己的独立 checkout，例如个人工作目录中的仓库副本；不要打开
   或改写现有实验、生产任务或其他协作者的 checkout。
5. 在该 checkout 内创建 `.venv`，安装 `.[dev]`，选择远端解释器后运行相同检查。

首次连接的 VS Code Server 安装和服务器使用由主机授权决定。本仓库不包含 SSH
连接配置，也不会启动训练或停止旧任务。详细步骤以
[VS Code Remote-SSH 官方文档](https://code.visualstudio.com/docs/remote/ssh)为准。

## Live Share 结对会话

Live Share 用于一个主持工作区的实时编辑、调试或经主持人许可的终端协作；它不是
Git 分支、PR、仓库写权限或服务器账号的替代品。

1. 参与者安装 **Live Share**，各自完成所需登录。
2. 主持人在自己的工作分支启动 **Live Share: Start Collaboration Session**，只向
   本次参与者发送会话链接。由主持人决定会话的编辑和终端访问范围。
3. 参与者通过会话链接加入，不交换 GitHub 密码、SSH 密码、私钥或访问令牌。
4. 主持人结束会话后检查所有修改和命令结果，再按正常流程提交自己的分支和 PR。

会话中的改动落在主持人的工作区，参与者不要在同一工作区各自切换 Git 分支。
启用会话前移开敏感文件，避免在共享终端输出凭据或私有数据。安装、登录和使用见
[Live Share 官方说明](https://learn.microsoft.com/en-us/visualstudio/liveshare/use/vscode)。

## 实验与公开材料

协作开发不自动授权服务器实验或数据发布。真实实验需单独冻结数据、配置、权重、
预算、角色边界和评价规则。示例与软件测试结果应标明其制造数据范围，不能写成
全湖精度或真实物理验收。仓库保留现有 MIT LICENSE；新增第三方代码和数据分别记录
许可与来源。
