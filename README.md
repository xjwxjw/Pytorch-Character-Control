## Common practice of homanoid character control with pytorch

## Work plan 
- [x] 熟悉键盘输入map到trajectory的逻辑，并实现Python版本的code
- [x] 分析现有数据的最大转向速度
- [ ] 对齐控制逻辑的最大转向速度
- [x] 将现有QuaterNet模型迁移到Avatar数据上
- [x] 将训好的模型和控制逻辑代码合并
- [ ] 增加loss constraint，保证逻辑状态与人物状态的一致性
- [ ] case by case的测试样例设计
- [ ] Visualization wit 3D mesh in UE4.
- [ ] Running pose implementation.
- [x] Bug fix in traj pred and 180 turn.
- [x] Balance betweend control logic and mocap data (Make it adjustable?).
- [ ] Foot sliding evaluation.
- [ ] Camera view implementation. 

## TODO List
- [ ] 实现skeleton的迁移以匹配当前的数据
- [ ] 实现模型从pytorch到C#的迁移
- [ ] 修改当前的控制逻辑以匹配当前的数据



## Work Status

[This sheet](https://docs.google.com/spreadsheets/d/1UhpiTP2QyN1eut8PT26ld14hbsh1qCkTzvO-jR4Dr7I/edit?usp=sharing)
