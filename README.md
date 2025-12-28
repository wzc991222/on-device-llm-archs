# 面向端侧设备的高效大语言模型架构探索
This repo documents my explorations in on-device LLM architecture design in 2025.
The following is a detailed description in Chinese.

本仓库记录了我在2025年对适合端侧设备的高效大语言模型架构的探索，包括三个方向：
1. 层间专家共享的混合专家模型：[sharedmoe](https://github.com/wzc991222/on-device-llm-archs/tree/main/on-device-llm-archs/sharedmoe)
2. 支持块内部分并行解码的大语言模型：[blocklatent](https://github.com/wzc991222/on-device-llm-archs/tree/main/on-device-llm-archs/blocklatent)
3. 基于块级路由的混合专家模型：[blockmoe](https://github.com/wzc991222/on-device-llm-archs/tree/main/on-device-llm-archs/blockmoe)
以上探索虽然暂未取得真正的成功，但也有很多收获。
由于时间关系，没有对代码文件做任何整理，直接将所有版本的未加注释与修改的模型代码全都上传了，因此本仓库暂时不具备可读性。
之后会写一篇文章详细讲述以上内容，并整理相应的代码。
